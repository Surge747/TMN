import os
import json
import pickle
import time
from typing import Sequence, Callable, Optional
import math # Need math for tan

# PyTorch related imports
import torch
import torch.nn as nn
# Note: We might need torch.optim, torch.utils.data later

# Standard numerical/image/utility libraries
import numpy # Keep numpy for compatibility and specific operations
import cv2   # OpenCV for image saving
from tqdm import tqdm # Progress bars
import matplotlib.pyplot as plt # For plotting (if needed later)
from PIL import Image # For image loading (if using original data loader structure)
from multiprocessing.pool import ThreadPool # For data loading (if using original structure)
import functools # For partial functions (might be used in evaluation)
import gc # Garbage collection

# --- Configuration ---
# User selects the scene type and object name
scene_type = "synthetic"      # OPTIONS: "synthetic", "forwardfacing", "real360"
object_name = "drums_resized_200" # EXAMPLE: replace with your desired scene name

# --- Construct Scene Directory Path ---
# Adjusted paths to likely locations
if scene_type == "synthetic":
    # Assuming synthetic data is in 'datasets/nerf_synthetic/...'
    scene_dir = os.path.join("datasets", object_name)
elif scene_type == "forwardfacing":
    # Assuming forwardfacing data is in 'datasets/nerf_llff_data/...'
    scene_dir = os.path.join("datasets", object_name)
elif scene_type == "real360":
    # Assuming real360 data is in 'datasets/nerf_real_360/...' (adjust if needed)
    scene_dir = os.path.join("datasets", object_name)
else:
    raise ValueError(f"Unknown scene_type: {scene_type}")

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Directory Setup ---
weights_dir = "weights"; samples_dir = "samples"
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# --- Image Saving Helper ---
# --- Image Saving Helper ---
def write_floatpoint_image(name: str, img_tensor: torch.Tensor):
  """ Saves a [0, 1] float Tensor image to an 8-bit file using OpenCV. """
  # Ensure input is a tensor before proceeding
  if not isinstance(img_tensor, torch.Tensor):
      # If input is somehow numpy, convert it back for consistent handling
      print(f"Warning: write_floatpoint_image received numpy array for {name}. Converting to tensor.")
      img_tensor = torch.from_numpy(numpy.array(img_tensor))

  # Ensure tensor is on CPU BEFORE converting to numpy
  if img_tensor.device != torch.device('cpu'):
      img_tensor_cpu = img_tensor.cpu()
  else:
      img_tensor_cpu = img_tensor

  # Detach if needed before numpy conversion
  if img_tensor_cpu.requires_grad:
      img_np = img_tensor_cpu.detach().numpy()
  else:
      img_np = img_tensor_cpu.numpy()

  # Scale, clip, convert type
  img_np = numpy.clip(img_np * 255.0, 0, 255).astype(numpy.uint8)

  # Handle grayscale / ensure 3 channels for saving
  if img_np.ndim == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
  elif img_np.shape[-1] == 1: img_np = cv2.cvtColor(img_np[..., 0], cv2.COLOR_GRAY2BGR)

  # Save with RGB -> BGR conversion
  cv2.imwrite(name, img_np[..., ::-1]) # Assumes input tensor/numpy was RGB

# --- Data Loading ---
print(f"Configuration: scene_type='{scene_type}', object_name='{object_name}', scene_dir='{scene_dir}'")

if scene_type == "synthetic": white_bkgd = True
else: white_bkgd = False # Covers forwardfacing and real360

# --- Pose Helper Functions (PyTorch Version, Device Aware) ---

def _normalize(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (norm + 1e-8)

def _viewmatrix(z: torch.Tensor, up: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    vec2 = _normalize(z); vec1_avg = up
    vec0 = _normalize(torch.cross(vec1_avg, vec2, dim=-1))
    vec1 = _normalize(torch.cross(vec2, vec0, dim=-1))
    m = torch.stack([vec0, vec1, vec2, pos], dim=-1)
    return m

def _poses_avg(poses: torch.Tensor) -> torch.Tensor:
    # Input poses assumed to be (N, 3, 5)
    hwf = poses[0:1, :3, 4:] # Keep dim 0, shape (1, 3, 1)
    center = poses[:, :3, 3].mean(dim=0)
    vec2 = _normalize(poses[:, :3, 2].sum(dim=0))
    up = poses[:, :3, 1].sum(dim=0)
    c2w_3x4 = _viewmatrix(vec2, up, center)
    # Ensure hwf is correctly broadcastable if needed later, return (3, 5)
    # If poses has N=1, hwf is (1,3,1), need to ensure consistent shape?
    # Original returned (3,5), let's stick to that.
    return torch.cat([c2w_3x4, hwf.squeeze(0)], dim=-1) # Should be (3, 5)

def _recenter_poses(poses: torch.Tensor) -> torch.Tensor:
    # Input poses assumed to be (N, 3, 5)
    poses_ = poses.clone()
    c2w_avg = _poses_avg(poses) # (3, 5)
    bottom_row = torch.tensor([[0, 0, 0, 1.]], dtype=poses.dtype, device=poses.device)
    c2w_avg_4x4 = torch.cat([c2w_avg[:3, :4], bottom_row], dim=0)
    bottom = bottom_row.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    # Use full 3x5 pose to get HWF info if needed, but transform 3x4 part
    poses_4x4 = torch.cat([poses[:, :3, :4], bottom], dim=-2)
    poses_recentered = torch.matmul(torch.linalg.inv(c2w_avg_4x4), poses_4x4)
    # Update only the 3x4 part, keeping the original 5th column (HWF)
    poses_[:, :3, :4] = poses_recentered[:, :3, :4]
    return poses_

def _transform_poses_pca(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Input poses assumed to be (N, 3, 5)
    poses_ = poses.clone()
    t = poses[:, :3, 3]; t_mean = t.mean(dim=0); t_centered = t - t_mean
    covariance = torch.matmul(t_centered.T, t_centered)
    eigval, eigvec = torch.linalg.eig(covariance); eigval=eigval.real; eigvec=eigvec.real
    inds = torch.argsort(eigval, descending=True); eigvec = eigvec[:, inds]; rot = eigvec.T
    if torch.linalg.det(rot) < 0:
      rot = torch.matmul(torch.diag(torch.tensor([1,1,-1],dtype=rot.dtype, device=rot.device)), rot)
    transform = torch.cat([rot, torch.matmul(rot, -t_mean.unsqueeze(1))], dim=1)
    bottom_row = torch.tensor([[0.,0.,0.,1.]], dtype=poses.dtype, device=poses.device)
    bottom = bottom_row.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    poses_4x4 = torch.cat([poses[:, :3, :4], bottom], dim=-2)
    transform_4x4 = torch.cat([transform, bottom_row], dim=0)
    poses_recentered = torch.matmul(transform_4x4, poses_4x4)
    poses_recentered_3x4 = poses_recentered[..., :3, :4]
    if poses_recentered_3x4.mean(dim=0)[2, 1] < 0:
        flip_mat = torch.diag(torch.tensor([1,-1,-1], dtype=poses.dtype, device=poses.device))
        poses_recentered_3x4 = torch.matmul(flip_mat, poses_recentered_3x4)
        flip_transform = torch.diag(torch.tensor([1,-1,-1,1], dtype=transform.dtype, device=transform.device))
        transform_4x4 = torch.matmul(flip_transform, transform_4x4)
    scale_factor = 1.0 / torch.max(torch.abs(poses_recentered_3x4[:, :3, 3]))
    poses_recentered_3x4[:, :3, 3] *= scale_factor
    scale_mat = torch.diag(torch.tensor([scale_factor]*3 + [1.0], dtype=transform.dtype, device=transform.device))
    transform_4x4 = torch.matmul(scale_mat, transform_4x4)
    # Update only the 3x4 part, keeping the original 5th column (HWF)
    poses_[:, :3, :4] = poses_recentered_3x4
    return poses_, transform_4x4

# --- Scene Type Specific Loaders (Refined) ---

if scene_type == "synthetic":
    def load_blender_pytorch(data_dir: str, split: str, device: torch.device = torch.device('cpu')) -> dict:
        with open(os.path.join(data_dir, f"transforms_{split}.json"), "r") as fp: meta = json.load(fp)
        cams_np = []; paths = []
        for frame in meta["frames"]:
            cams_np.append(numpy.array(frame["transform_matrix"], dtype=numpy.float32))
            paths.append(os.path.join(data_dir, frame["file_path"] + ".png"))
        def image_read_fn(fname):
            with open(fname, "rb") as imgin: image = numpy.array(Image.open(imgin).convert('RGBA'), dtype=numpy.float32)/255.
            return image
        with ThreadPool() as pool: images_np = pool.map(image_read_fn, paths)
        images_np = numpy.stack(images_np, axis=0) # N, H, W, 4
        images = torch.from_numpy(images_np).to(device)
        poses = torch.from_numpy(numpy.stack(cams_np, axis=0)).to(device) # N, 4, 4

        # *** Apply background color logic matching original JAX code ***
        if white_bkgd:
            alpha = images[..., 3:] # N, H, W, 1
            rgb = images[..., :3]   # N, H, W, 3
            images = rgb * alpha + (1.0 - alpha) # Blend onto white
        else:
            # Alpha blend onto black
            images = images[..., :3] * images[..., 3:]

        h, w = images.shape[1:3]; camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * w / math.tan(0.5 * camera_angle_x)
        hwf = torch.tensor([h, w, focal], dtype=torch.float32, device=device)

        # Return c2w as 3x4, but also the full 4x4 pose and original HWF
        return {'images': images, 'c2w': poses[:, :3, :4], 'hwf': hwf, 'poses_all': poses}

    print("Loading synthetic data...")
    data = {'train' : load_blender_pytorch(scene_dir, 'train', device=device),
            'test' : load_blender_pytorch(scene_dir, 'test', device=device)}
    print("Data loading complete.")

elif scene_type == "forwardfacing" or scene_type == "real360":
    def load_LLFF_pytorch(data_dir: str, split: str, factor: int = 4, llffhold: int = 8, device: torch.device = torch.device('cpu')) -> dict:
        imgdir_suffix = f"_{factor}" if factor > 0 else ""
        imgdir = os.path.join(data_dir, "images" + imgdir_suffix)
        if not os.path.exists(imgdir): raise ValueError(f"Image folder {imgdir} doesn't exist.")
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.lower().endswith(("jpg", "png"))]
        def image_read_fn(fname):
            with open(fname, "rb") as imgin: image = numpy.array(Image.open(imgin), dtype=numpy.float32) / 255.
            return image
        with ThreadPool() as pool: images_np = pool.map(image_read_fn, imgfiles)
        images_np = numpy.stack(images_np, axis=-1) # H, W, C, N
        with open(os.path.join(data_dir, "poses_bounds.npy"), "rb") as fp: poses_arr = numpy.load(fp)
        poses_np = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0]); bds_np = poses_arr[:, -2:].transpose([1, 0])
        if poses_np.shape[-1] != images_np.shape[-1]: raise RuntimeError("Mismatch image/pose count")
        poses_np[:2, 4, :] = numpy.array(images_np.shape[:2]).reshape([2, 1])
        poses_np[2, 4, :] = poses_np[2, 4, :] * 1. / factor
        poses_np = numpy.concatenate([poses_np[:, 1:2, :], -poses_np[:, 0:1, :], poses_np[:, 2:, :]], 1)
        poses_np = numpy.moveaxis(poses_np, -1, 0).astype(numpy.float32) # N, 3, 5
        images_np = numpy.moveaxis(images_np, -1, 0).astype(numpy.float32); bds_np = numpy.moveaxis(bds_np, -1, 0).astype(numpy.float32)

        poses = torch.from_numpy(poses_np).to(device)
        bds = torch.from_numpy(bds_np).to(device)

        if scene_type == "real360": poses, _ = _transform_poses_pca(poses)
        elif scene_type == "forwardfacing":
            scale = 1.0 / (torch.min(bds) * 0.75)
            poses[:, :3, 3] *= scale; bds *= scale
            poses = _recenter_poses(poses)

        num_frames = poses.shape[0]; i_all = numpy.arange(num_frames)
        i_test = i_all[::llffhold]; i_train = numpy.array([i for i in i_all if i not in i_test])
        indices = i_train if split == "train" else i_test

        images = torch.from_numpy(images_np[indices]).to(device)
        poses = poses[indices] # Shape (N_split, 3, 5)
        bds = bds[indices] # Shape (N_split, 2)

        camtoworlds = poses[:, :3, :4] # N_split, 3, 4
        # Extract H, W, F from the 5th column of the first pose in the split
        h, w, focal = poses[0, 0, 4], poses[0, 1, 4], poses[0, 2, 4]
        hwf = torch.tensor([h, w, focal], dtype=torch.float32, device=device)

        # *** Return full poses and bounds as well ***
        return {'images': images, 'c2w': camtoworlds, 'hwf': hwf, 'poses': poses, 'bds': bds}

    print("Loading LLFF/Real360 data...")
    data = {'train' : load_LLFF_pytorch(scene_dir, 'train', device=device),
            'test' : load_LLFF_pytorch(scene_dir, 'test', device=device)}
    print("Data loading complete.")

# --- Final Verification and Setup ---
splits = ['train', 'test']
for s in splits:
    print(f"\n{s.capitalize()} Data:")
    if s in data and data[s] is not None:
        for k, v in data[s].items(): # Iterate through dictionary items
            print(f"  {k}: {v.shape}, Device: {v.device}")
    else: print(f"  {s} data not loaded for scene_type {scene_type}")

if 'train' in data and data['train'] is not None:
    images = data['train']['images']
    # Use 'c2w' for standard camera-to-world if that's what downstream needs
    poses_for_plot = data['train']['c2w']
    hwf = data['train']['hwf']

    print("\nSaving sample image...")
    write_floatpoint_image(os.path.join(samples_dir, "training_image_sample.png"), images[0])

    print("Plotting camera positions...")
    poses_np = poses_for_plot.cpu().numpy() # Use c2w (3x4) for plotting positions
    for i in range(min(3, poses_np.shape[1])):
        # Plotting translation part (last column)
         if poses_np.shape[1] > (i + 1) % 3: # Check second axis exists
            plt.figure()
            # Scatter plot requires numpy arrays
            x_coords = poses_np[:, i, 3]
            y_coords = poses_np[:, (i + 1) % 3, 3]
            plt.scatter(x_coords, y_coords)
            plt.xlabel(f"Axis {i}"); plt.ylabel(f"Axis {(i+1)%3}")
            plt.title(f"Camera Positions (Axes {i} vs {(i+1)%3})")
            plt.axis('equal'); plt.grid(True)
            plt.savefig(os.path.join(samples_dir, f"training_camera_{i}.png"))
            plt.close()

    bg_color = torch.mean(images.float()).item()
    print(f"Calculated background color: {bg_color:.4f}")
else:
    print("\nSkipping image save/plot/bg_color calculation as training data not loaded.")
    bg_color = 0.5

# #%% --------------------------------------------------------------------------------
# # ## Helper functions <<< Block Start Marker
# #%%                      (Previous code ends here)

# --- PyTorch Helper Functions ---

# Adam optimizer parameters (will be used when creating optimizer)
adam_kwargs = {
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-15, # Epsilon for numerical stability in Adam
}

# Seed for reproducibility (optional but recommended)
# Set seeds at the beginning of your script execution
global_seed = 1
torch.manual_seed(global_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(global_seed)
# Note: Randomness for batch sampling might need separate handling if using DataLoader workers

# General math functions.
# Note: matmul does not need explicit precision in torch like in JAX
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Wrapper for torch.matmul for clarity."""
  return torch.matmul(a, b)

# _normalize function already defined during data loading conversion
# def _normalize(x: torch.Tensor) -> torch.Tensor: ...

def sinusoidal_encoding(position: torch.Tensor, minimum_frequency_power: int,
                        maximum_frequency_power: int, include_identity: bool = False) -> torch.Tensor:
  """Computes sinusoidal positional encoding for PyTorch Tensors."""
  # Input: position (..., D)
  position = position.float()
  # Frequencies: (F,)
  frequency = 2.0**torch.arange(minimum_frequency_power, maximum_frequency_power,
                                 dtype=torch.float32, device=position.device)
  # Angles: (..., D, F)
  angle = position.unsqueeze(-1) * frequency.unsqueeze(0) # Correct broadcasting
  # Encodings: (..., D, 2*F) -> (..., D*2*F)
  # Original JAX flattened feature and frequency dimensions.
  # Order matters: [sin(f0), cos(f0), sin(f1), cos(f1), ...]
  encoding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) # ..., D, 2*F
  encoding = encoding.reshape(*position.shape[:-1], -1) # Flatten last two dims -> ..., D*2*F

  if include_identity:
    encoding = torch.cat([position, encoding], dim=-1)
  return encoding

# Pose/ray math.
def generate_rays(pixel_coords: torch.Tensor, pix2cam: torch.Tensor, cam2world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Generate camera rays from pixel coordinates and poses (PyTorch)."""
  # pixel_coords: (..., 2) e.g., (H, W, 2) or (N, 2)
  # pix2cam: (3, 3)
  # cam2world: (..., 3, 4) e.g., (3, 4) or (N, 3, 4) matching leading dims of pixel_coords

  # Add homogeneous coordinate
  homog = torch.ones_like(pixel_coords[..., :1])
  # Center pixel coordinates and add homogeneous coord
  pixel_dirs_uv = torch.cat([pixel_coords + 0.5, homog], dim=-1) # (..., 3)

  # Transform pixel coordinates to camera space directions
  # (..., 3) -> (..., 3, 1) for matmul
  pixel_dirs_uv = pixel_dirs_uv.unsqueeze(-1)
  # pix2cam @ pixel_dirs_uv : (3, 3) @ (..., 3, 1) -> (..., 3, 1)
  # Use broadcasting: unsqueeze pix2cam if pixel_dirs_uv has leading dims
  cam_dirs = torch.matmul(pix2cam, pixel_dirs_uv) # Broadcasting handles leading dims

  # Transform camera space directions to world space directions
  # Use only rotation part of cam2world: (..., 3, 3)
  rot = cam2world[..., :3, :3]
  # rot @ cam_dirs : (..., 3, 3) @ (..., 3, 1) -> (..., 3, 1)
  ray_dirs = torch.matmul(rot, cam_dirs).squeeze(-1) # (..., 3)

  # World space ray origin is the camera center
  # Broadcast origin to match ray directions shape
  ray_origins = torch.broadcast_to(cam2world[..., :3, 3], ray_dirs.shape) # (..., 3)

  return ray_origins, ray_dirs

def pix2cam_matrix(height: float, width: float, focal: float, device: torch.device) -> torch.Tensor:
  """Inverse intrinsic matrix for a pinhole camera (PyTorch Tensor)."""
  if isinstance(height, torch.Tensor): height = height.item()
  if isinstance(width, torch.Tensor): width = width.item()
  if isinstance(focal, torch.Tensor): focal = focal.item()
  return  torch.tensor([
      [1./focal, 0, -.5 * width / focal],
      [0, -1./focal, .5 * height / focal],
      [0, 0, -1.],
  ], dtype=torch.float32, device=device)

def camera_ray_batch(cam2world: torch.Tensor, hwf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Generate rays for a pinhole camera (full image) (PyTorch)."""
  # cam2world: (3, 4) or (4, 4) -> use first (3, 4)
  # hwf: (3,) [H, W, F]
  height, width, focal = hwf[0].item(), hwf[1].item(), hwf[2].item()
  pix2cam = pix2cam_matrix(height, width, focal, device=cam2world.device)
  # Create pixel coordinate grid (H, W, 2)
  y_coords, x_coords = torch.meshgrid(torch.arange(height, device=cam2world.device),
                                      torch.arange(width, device=cam2world.device),
                                      indexing='ij') # H, W tensors
  pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float() # H, W, 2
  # generate_rays handles broadcasting cam2world to all pixels
  return generate_rays(pixel_coords, pix2cam, cam2world[:3,:4])

def random_ray_batch(batch_size: int, data: dict, device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
  """Generate a random batch of ray data (PyTorch)."""
  # Assumes data tensors are already on the correct device
  num_cams, height, width, _ = data['images'].shape
  num_poses = data['c2w'].shape[0]
  if num_cams != num_poses:
      print(f"Warning: Mismatch in image count ({num_cams}) and pose count ({num_poses})")
      num_cams = min(num_cams, num_poses) # Use the minimum count

  # Randomly sample indices using torch.randint
  cam_ind = torch.randint(0, num_cams, (batch_size,), device=device)
  y_ind = torch.randint(0, height, (batch_size,), device=device)
  x_ind = torch.randint(0, width, (batch_size,), device=device)

  # Combine coordinates (batch_size, 2)
  pixel_coords = torch.stack([x_ind, y_ind], dim=-1).float()

  # Get intrinsics matrix (H, W, F are on device)
  pix2cam = pix2cam_matrix(data['hwf'][0], data['hwf'][1], data['hwf'][2], device=device)

  # Get corresponding camera poses
  cam2world = data['c2w'][cam_ind] # (batch_size, 3, 4)

  # Generate rays - generate_rays handles batch dimension N=batch_size
  rays = generate_rays(pixel_coords, pix2cam, cam2world) # (origins, directions)

  # Get corresponding ground truth pixel colors
  pixels = data['images'][cam_ind, y_ind, x_ind] # (batch_size, C)

  return rays, pixels

# Learning rate helpers.
def log_lerp(t: float, v0: float, v1: float) -> float:
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0: raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = math.log(v0); lv1 = math.log(v1) # Use standard math log/exp for scalars
  t = max(0.0, min(1.0, t)) # Clamp t
  lerped_log = t * (lv1 - lv0) + lv0
  return math.exp(lerped_log)

def lr_fn(step: int, max_steps: int, lr0: float, lr1: float, lr_delay_steps: int = 20000, lr_delay_mult: float = 0.1) -> float:
  """Calculates learning rate with optional warmup and log-linear decay."""
  if lr_delay_steps > 0:
    # Use standard math sin
    delay_progress = max(0.0, min(1.0, step / lr_delay_steps))
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(0.5 * math.pi * delay_progress)
  else:
    delay_rate = 1.0
  log_lerp_factor = step / max_steps
  current_lr = delay_rate * log_lerp(log_lerp_factor, lr0, lr1)
  return current_lr


# #%% --------------------------------------------------------------------------------
# # ## Plane parameters and setup <<< Block Start Marker
# #%%

# Grid size configuration
point_grid_size = 128
# Scaling factor for point_grid offset learning rate (relative to main LR)
point_grid_diff_lr_scale = 16.0 / point_grid_size # Keep as float

# --- Scene Type Specific Parameters ---
# Initialize variables to be defined within the conditional blocks
grid_min: Optional[torch.Tensor] = None
grid_max: Optional[torch.Tensor] = None
scene_grid_zmax: Optional[float] = None
scene_grid_zcc: Optional[float] = None
get_taper_coord: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
inverse_taper_coord: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

if scene_type=="synthetic":
  scene_grid_scale = 1.2
  if object_name in ["hotdog", "mic", "ship"]: scene_grid_scale = 1.5
  grid_min_np = numpy.array([-1, -1, -1]) * scene_grid_scale
  grid_max_np = numpy.array([ 1,  1,  1]) * scene_grid_scale

  def get_taper_coord_synthetic(p: torch.Tensor) -> torch.Tensor: return p
  def inverse_taper_coord_synthetic(p: torch.Tensor) -> torch.Tensor: return p
  get_taper_coord = get_taper_coord_synthetic
  inverse_taper_coord = inverse_taper_coord_synthetic

elif scene_type=="forwardfacing":
  scene_grid_taper = 1.25; scene_grid_zstart = 25.0; scene_grid_zend = 1.0
  scene_grid_scale = 0.7
  grid_min_np = numpy.array([-scene_grid_scale, -scene_grid_scale, 0])
  grid_max_np = numpy.array([ scene_grid_scale,  scene_grid_scale, 1])
  log_z_start = math.log(scene_grid_zstart); log_z_end = math.log(scene_grid_zend)
  log_z_diff = log_z_start - log_z_end

  def get_taper_coord_ff(p: torch.Tensor) -> torch.Tensor:
    pz = torch.clamp(-p[..., 2:3], min=1e-10)
    px = p[..., 0:1] / (pz * scene_grid_taper)
    py = p[..., 1:2] / (pz * scene_grid_taper)
    log_pz = torch.log(pz)
    pz_tapered = (log_pz - log_z_end) / log_z_diff
    return torch.cat([px, py, pz_tapered], dim=-1)

  def inverse_taper_coord_ff(p_tapered: torch.Tensor) -> torch.Tensor:
    log_pz = p_tapered[..., 2:3] * log_z_diff + log_z_end
    pz = torch.exp(log_pz)
    px = p_tapered[..., 0:1] * (pz * scene_grid_taper)
    py = p_tapered[..., 1:2] * (pz * scene_grid_taper)
    pz_world = -pz
    return torch.cat([px, py, pz_world], dim=-1)
  get_taper_coord = get_taper_coord_ff
  inverse_taper_coord = inverse_taper_coord_ff

elif scene_type=="real360":
  scene_grid_zmax = 16.0
  if object_name == "gardenvase": scene_grid_zmax = 9.0
  grid_min_np = numpy.array([-1, -1, -1]); grid_max_np = numpy.array([ 1,  1,  1])

  def get_taper_coord_real360(p: torch.Tensor) -> torch.Tensor: return p
  def inverse_taper_coord_real360(p: torch.Tensor) -> torch.Tensor: return p
  get_taper_coord = get_taper_coord_real360
  inverse_taper_coord = inverse_taper_coord_real360

  scene_grid_zcc = -1.0
  # Use numpy for this one-time calculation
  for i in range(10000):
      j = numpy.log(scene_grid_zmax) + i / 1000.0
      if numpy.exp(j) - scene_grid_zmax * j + (j - 1) > 0:
          scene_grid_zcc = float(j)
          break
  if scene_grid_zcc < 0: raise RuntimeError("Failed to compute scene_grid_zcc")
  print(f"Calculated scene_grid_zcc: {scene_grid_zcc:.4f}")

# Assert that parameters were set
assert grid_min_np is not None and grid_max_np is not None, "Grid bounds not set"
assert get_taper_coord is not None and inverse_taper_coord is not None, "Taper functions not set"

# Convert Grid Bounds to Tensors (moved after conditional blocks)
grid_min = torch.from_numpy(grid_min_np).float().to(device)
grid_max = torch.from_numpy(grid_max_np).float().to(device)
grid_range = grid_max - grid_min # Tensor on device

# Grid Initialization (Placeholders - to be done in nn.Module)
# grid_dtype = torch.float32 # Defined by tensor creation later

# --- Grid Helper Functions (PyTorch Version) ---
# Make grid_min, grid_max, point_grid_size parameters to this function
def get_acc_grid_masks(taper_positions: torch.Tensor, acc_grid: torch.Tensor,
                       grid_min: torch.Tensor, grid_max: torch.Tensor,
                       point_grid_size: int) -> torch.Tensor:
    """Samples the acc_grid at given positions (PyTorch version)."""
    grid_range = grid_max - grid_min # Calculate inside function

    # Map tapered positions to grid coordinates [0, G-1]
    grid_positions_float = (taper_positions - grid_min) * (point_grid_size / grid_range)

    # Boundary check constants
    min_bound = 1.0 - 1e-6
    max_bound = float(point_grid_size - 1) - 1e-6

    # Create mask for points within the valid *interior* index range [1, G-2]
    grid_masks = (grid_positions_float[..., 0] >= min_bound) & (grid_positions_float[..., 0] < max_bound) & \
                 (grid_positions_float[..., 1] >= min_bound) & (grid_positions_float[..., 1] < max_bound) & \
                 (grid_positions_float[..., 2] >= min_bound) & (grid_positions_float[..., 2] < max_bound)

    # Clamp grid positions to [0, G-1] before casting to long for indexing
    grid_indices = torch.clamp(grid_positions_float, 0, point_grid_size - 1.0 - 1e-6)
    grid_indices = grid_indices.long() # Use long for indexing

    # Sample acc_grid using integer indices (effectively nearest neighbor)
    # Ensure acc_grid has the expected shape (G, G, G) when passed in
    acc_grid_values = acc_grid[grid_indices[..., 0], grid_indices[..., 1], grid_indices[..., 2]]

    # Apply the boundary mask (zero out values derived from outside [1, G-2])
    acc_grid_values = acc_grid_values * grid_masks.float()

    return acc_grid_values







# --- Ray Intersection and Geometry Functions (PyTorch) ---

# Helper to avoid division by zero
def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # Add epsilon to the denominator, potentially adjusting by sign to avoid crossing zero
    # A simpler approach is just adding a small positive epsilon if b is guaranteed non-negative
    # Or handle potential sign crossing:
    return a / (b + torch.sign(b) * eps + eps) # Adjust epsilon based on sign


# --- gridcell_from_rays (Functions for each scene type) ---

def gridcell_from_rays_synthetic(
    rays: tuple[torch.Tensor, torch.Tensor],
    acc_grid: torch.Tensor,
    keep_num: int,
    threshold: float,
    point_grid_size: int,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds intersections with undeformed grid planes for synthetic scenes."""
    ray_origins, ray_directions = rays
    dtype = ray_origins.dtype; device = ray_origins.device
    batch_shape = ray_origins.shape[:-1]
    num_batch_dims = len(batch_shape)
    small_step = 1e-5; epsilon = 1e-5; inf_val = 1e10 # Use a large float for infinity
    grid_range = grid_max - grid_min

    # Unsqueeze rays for broadcasting against planes: shape (..., 1, 3)
    ox, oy, oz = ray_origins[..., 0:1].unsqueeze(-2), ray_origins[..., 1:2].unsqueeze(-2), ray_origins[..., 2:3].unsqueeze(-2)
    dx, dy, dz = ray_directions[..., 0:1].unsqueeze(-2), ray_directions[..., 1:2].unsqueeze(-2), ray_directions[..., 2:3].unsqueeze(-2)

    # Masks for axis-aligned rays
    dxm = (torch.abs(dx) < epsilon) # (..., 1, 3)
    dym = (torch.abs(dy) < epsilon)
    dzm = (torch.abs(dz) < epsilon)

    # Safe denominators
    safe_dx = dx + dxm.type_as(dx) * epsilon
    safe_dy = dy + dym.type_as(dy) * epsilon
    safe_dz = dz + dzm.type_as(dz) * epsilon

    # Grid plane coordinates [0, 1] -> world coordinates
    layers_norm = torch.arange(point_grid_size + 1, dtype=dtype, device=device) / point_grid_size
    # Reshape for broadcasting: (1,...,1, G+1, 1)
    layers_norm = layers_norm.view(*([1] * num_batch_dims), point_grid_size + 1, 1)

    planes_x = layers_norm * grid_range[0] + grid_min[0]
    planes_y = layers_norm * grid_range[1] + grid_min[1]
    planes_z = layers_norm * grid_range[2] + grid_min[2]

    # Calculate intersection distances (t) for each axis separately
    # Ensure broadcasting aligns correctly:
    # planes_x (..., G+1, 1) - ox[..., 0:1] (..., 1, 1) => result (..., G+1, 1)
    # safe_dx[..., 0:1] (..., 1, 1)
    tx = (planes_x - ox[..., 0:1]) / safe_dx[..., 0:1] # (..., G+1, 1)
    ty = (planes_y - oy[..., 0:1]) / safe_dy[..., 0:1] # (..., G+1, 1)
    tz = (planes_z - oz[..., 0:1]) / safe_dz[..., 0:1] # (..., G+1, 1)

    # Handle axis-aligned rays
    # dxm shape is (..., 1, 3). Unsqueeze mask component to (..., 1, 1) for broadcasting
    dxm_b = dxm[..., 0].unsqueeze(-1) # Shape (..., 1, 1)
    dym_b = dym[..., 0].unsqueeze(-1) # Shape (..., 1, 1)
    dzm_b = dzm[..., 0].unsqueeze(-1) # Shape (..., 1, 1)

    # Apply where condition (Shapes: tx(..., G+1, 1), dxm_b(..., 1, 1)) -> Broadcasts correctly
    tx = torch.where(dxm_b, inf_val, tx)
    ty = torch.where(dym_b, inf_val, ty)
    tz = torch.where(dzm_b, inf_val, tz)

    # Concatenate along the *last* dimension now: (..., G+1, 3)
    txyz_cat = torch.cat([tx, ty, tz], dim=-1)
    # Reshape to flatten plane and axis dimension: (..., (G+1)*3)
    txyz = txyz_cat.reshape(*batch_shape, -1)

    # Mask intersections behind origin
    txyz = torch.where(txyz <= 0, inf_val, txyz)

    # --- Empty Space Skipping using acc_grid ---
    txyz_shifted = txyz + small_step # Avoid landing exactly on plane
    # world_positions: (..., 1, 3) + (..., 1, 3) * (..., N_intersections, 1) -> (..., N_intersections, 3)
    world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * txyz_shifted.unsqueeze(-1)

    # Get occupancy values (function assumes taper_positions=world_positions for synthetic)
    acc_grid_values = get_acc_grid_masks(world_positions, acc_grid, grid_min, grid_max, point_grid_size)

    # Mask out intersections in low-occupancy cells
    txyz = torch.where(acc_grid_values < threshold, inf_val, txyz_shifted - small_step) # Revert shift for valid t

    # Sort and keep top k intersections
    txyz_sorted, _ = torch.sort(txyz, dim=-1)
    t_kept = txyz_sorted[..., :keep_num] # (..., keep_num)

    # --- Calculate grid indices for kept points ---
    world_positions_kept = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_kept.unsqueeze(-1)
    grid_positions_float = (world_positions_kept - grid_min) * (point_grid_size / grid_range)

    # Mask for valid interior indices [1, G-2]
    min_bound = 0.0 - 1e-6         # allow >= 0
    max_bound = float(point_grid_size) - 1e-6  # allow < G
    grid_masks = (grid_positions_float[..., 0] >= min_bound) & (grid_positions_float[..., 0] < max_bound) & \
                 (grid_positions_float[..., 1] >= min_bound) & (grid_positions_float[..., 1] < max_bound) & \
                 (grid_positions_float[..., 2] >= min_bound) & (grid_positions_float[..., 2] < max_bound)

    # Clamp indices to [0, G-1] and cast to long for safe indexing
    grid_indices = torch.clamp(grid_positions_float, 0, point_grid_size - 1.0 - 1e-6)
    grid_indices = grid_indices.long()

    return grid_indices, grid_masks


def gridcell_from_rays_forwardfacing(
    rays: tuple[torch.Tensor, torch.Tensor],
    acc_grid: torch.Tensor,
    keep_num: int,
    threshold: float,
    point_grid_size: int,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    # Add FF specific params
    scene_grid_taper: float,
    log_z_start: float,
    log_z_end: float,
    log_z_diff: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds intersections with tapered grid planes for forward-facing scenes."""
    ray_origins, ray_directions = rays
    dtype = ray_origins.dtype; device = ray_origins.device
    batch_shape = ray_origins.shape[:-1]; num_batch_dims = len(batch_shape)
    small_step = 1e-5; epsilon = 1e-10; inf_val = 1e10
    grid_range = grid_max - grid_min

    # Unsqueeze rays for broadcasting: (..., 1, 3)
    ox, oy, oz = ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2)
    dx, dy, dz = ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2)

    # Layers in normalized grid space [0, 1] -> (..., G+1, 1)
    layers_norm = torch.arange(point_grid_size + 1, dtype=dtype, device=device) / point_grid_size
    layers_norm = layers_norm.view(*([1] * num_batch_dims), point_grid_size + 1, 1)

    # Calculate intersections with Z planes (defined by log-z)
    log_pz = layers_norm.squeeze(-1) * log_z_diff + log_z_end # Shape (..., G+1)
    Zlayers = -torch.exp(log_pz) # World Z coords (..., G+1)
    dzm = (torch.abs(dz) < epsilon)
    safe_dz = dz + dzm.type_as(dz) * epsilon
    tz = (Zlayers.unsqueeze(-1) - oz) / safe_dz # Calculate t (..., G+1, 1)
    tz = torch.where(dzm.squeeze(-1), inf_val, tz) # Apply mask (..., G+1, 1)

    # Calculate intersections with X planes (defined by x/z = const)
    Xlayers_t = layers_norm * grid_range[0] + grid_min[0] # Tapered X coords (..., G+1, 1)
    Xlayers_ = Xlayers_t * scene_grid_taper # x/z ratio factor
    dxx = dx - Xlayers_ * dz
    dxm_taper = (torch.abs(dxx) < epsilon)
    safe_dxx = dxx + dxm_taper.type_as(dxx) * epsilon
    tx = (oz * Xlayers_ - ox) / safe_dxx # Calculate t (..., G+1, 1)
    tx = torch.where(dxm_taper.squeeze(-1), inf_val, tx) # Apply mask (..., G+1, 1)

    # Calculate intersections with Y planes (defined by y/z = const)
    Ylayers_t = layers_norm * grid_range[1] + grid_min[1] # Tapered Y coords (..., G+1, 1)
    Ylayers_ = Ylayers_t * scene_grid_taper # y/z ratio factor
    dyy = dy - Ylayers_ * dz
    dym_taper = (torch.abs(dyy) < epsilon)
    safe_dyy = dyy + dym_taper.type_as(dyy) * epsilon
    ty = (oz * Ylayers_ - oy) / safe_dyy # Calculate t (..., G+1, 1)
    ty = torch.where(dym_taper.squeeze(-1), inf_val, ty) # Apply mask (..., G+1, 1)

    # Concatenate and mask t values
    txyz = torch.cat([tx, ty, tz], dim=-1).view(*batch_shape, -1) # (..., (G+1)*3)
    txyz = torch.where(txyz <= 0, inf_val, txyz)

    # --- Empty Space Skipping (using tapered coords) ---
    txyz_shifted = txyz + small_step
    world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * txyz_shifted.unsqueeze(-1)
    # Must use the correct taper function here
    taper_positions = get_taper_coord(world_positions)
    acc_grid_values = get_acc_grid_masks(taper_positions, acc_grid, grid_min, grid_max, point_grid_size)
    txyz = torch.where(acc_grid_values < threshold, inf_val, txyz_shifted - small_step)

    # Sort and keep top k
    txyz_sorted, _ = torch.sort(txyz, dim=-1)
    t_kept = txyz_sorted[..., :keep_num]

    # --- Calculate final grid indices ---
    world_positions_kept = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_kept.unsqueeze(-1)
    taper_positions_kept = get_taper_coord(world_positions_kept)
    grid_positions_float = (taper_positions_kept - grid_min) * (point_grid_size / grid_range)

    min_bound = 0.0 - 1e-6; max_bound = float(point_grid_size) - 1e-6
    grid_masks = (grid_positions_float[..., 0] >= min_bound) & (grid_positions_float[..., 0] < max_bound) & \
                 (grid_positions_float[..., 1] >= min_bound) & (grid_positions_float[..., 1] < max_bound) & \
                 (grid_positions_float[..., 2] >= min_bound) & (grid_positions_float[..., 2] < max_bound)

    grid_indices = torch.clamp(grid_positions_float, 0, point_grid_size - 1.0 - 1e-6).long()
    return grid_indices, grid_masks


def gridcell_from_rays_real360(
    rays: tuple[torch.Tensor, torch.Tensor],
    acc_grid: torch.Tensor,
    keep_num: int,
    threshold: float,
    point_grid_size: int,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds intersections with undeformed grid planes for real360 scenes (with near clip)."""
    ray_origins, ray_directions = rays
    dtype = ray_origins.dtype; device = ray_origins.device
    batch_shape = ray_origins.shape[:-1]; num_batch_dims = len(batch_shape)
    small_step = 1e-5; epsilon = 1e-5; inf_val = 1e10
    grid_range = grid_max - grid_min

    ox, oy, oz = ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2)
    dx, dy, dz = ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2)

    dxm = (torch.abs(dx) < epsilon); dym = (torch.abs(dy) < epsilon); dzm = (torch.abs(dz) < epsilon)
    safe_dx = dx + dxm.type_as(dx) * epsilon; safe_dy = dy + dym.type_as(dy) * epsilon; safe_dz = dz + dzm.type_as(dz) * epsilon

    layers_norm = torch.arange(point_grid_size + 1, dtype=dtype, device=device) / point_grid_size
    layers_norm = layers_norm.view(*([1] * num_batch_dims), point_grid_size + 1, 1)
    planes_x = layers_norm * grid_range[0] + grid_min[0]
    planes_y = layers_norm * grid_range[1] + grid_min[1]
    planes_z = layers_norm * grid_range[2] + grid_min[2]

    tx = (planes_x - ox[..., 0:1]) / safe_dx[..., 0:1]
    ty = (planes_y - oy[..., 0:1]) / safe_dy[..., 0:1]
    tz = (planes_z - oz[..., 0:1]) / safe_dz[..., 0:1]

    tx = torch.where(dxm[..., 0], inf_val, tx); ty = torch.where(dym[..., 0], inf_val, ty); tz = torch.where(dzm[..., 0], inf_val, tz)

    txyz = torch.cat([tx, ty, tz], dim=-1).view(*batch_shape, -1)

    # *** Near Clip specific to Real360 ***
    txyz = torch.where(txyz <= 0.2, inf_val, txyz) # Mask t <= 0.2

    # --- Empty Space Skipping ---
    txyz_shifted = txyz + small_step
    world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * txyz_shifted.unsqueeze(-1)
    # Real360 uses identity taper inside the [-1,1] cube for get_acc_grid_masks
    acc_grid_values = get_acc_grid_masks(world_positions, acc_grid, grid_min, grid_max, point_grid_size)
    txyz = torch.where(acc_grid_values < threshold, inf_val, txyz_shifted - small_step)

    # Sort and keep
    txyz_sorted, _ = torch.sort(txyz, dim=-1)
    t_kept = txyz_sorted[..., :keep_num]

    # --- Calculate grid indices ---
    world_positions_kept = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t_kept.unsqueeze(-1)
    grid_positions_float = (world_positions_kept - grid_min) * (point_grid_size / grid_range)

    min_bound = 0.0 - 1e-6; max_bound = float(point_grid_size) - 1e-6
    grid_masks = (grid_positions_float[..., 0] >= min_bound) & (grid_positions_float[..., 0] < max_bound) & \
                 (grid_positions_float[..., 1] >= min_bound) & (grid_positions_float[..., 1] < max_bound) & \
                 (grid_positions_float[..., 2] >= min_bound) & (grid_positions_float[..., 2] < max_bound)

    grid_indices = torch.clamp(grid_positions_float, 0, point_grid_size - 1.0 - 1e-6).long()
    return grid_indices, grid_masks

# Assign the correct function based on scene_type
if scene_type == "synthetic": gridcell_from_rays = gridcell_from_rays_synthetic
elif scene_type == "forwardfacing": gridcell_from_rays = gridcell_from_rays_forwardfacing
elif scene_type == "real360": gridcell_from_rays = gridcell_from_rays_real360


# --- Barycentric Coordinates ---
def get_barycentric(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
                    O: torch.Tensor, d: torch.Tensor,
                    eps: float = 1e-10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moller-Trumbore Ray-Triangle Intersection (PyTorch)."""
    # p1, p2, p3: Triangle vertices (..., 3)
    # O, d: Ray origin and direction (..., 3)

    # Edge vectors
    edge1 = p2 - p1 # (..., 3)
    edge2 = p3 - p1 # (..., 3)

    # Calculate determinant components
    pvec = torch.cross(d, edge2, dim=-1) # (..., 3)
    det = torch.sum(edge1 * pvec, dim=-1) # (...)

    # Check if determinant is near zero (ray parallel to triangle)
    det_mask = torch.abs(det) < eps

    # Inverse determinant (handle near-zero determinant)
    inv_det = 1.0 / (det + det_mask.type_as(det) * eps) # Add epsilon where det is near zero

    # Distance from vertex p1 to ray origin
    tvec = O - p1 # (..., 3)

    # Calculate u coordinate
    u = torch.sum(tvec * pvec, dim=-1) * inv_det # (...)

    # Calculate v coordinate
    qvec = torch.cross(tvec, edge1, dim=-1) # (..., 3)
    v = torch.sum(d * qvec, dim=-1) * inv_det # (...)

    # Calculate t (distance along ray) - not strictly needed for barycentric, but often calculated
    # t_intersect = torch.sum(edge2 * qvec, dim=-1) * inv_det

    # Check if intersection is within triangle bounds (u>=0, v>=0, u+v<=1)
    mask_bary = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Combine with determinant mask
    valid_mask = mask_bary & (~det_mask)

    # Calculate barycentric coordinates (a, b, c) = (1-u-v, u, v) -> matching P = a*p1 + b*p2 + c*p3
    # Or more commonly P = p1 + u*edge1 + v*edge2 -> P = (1-u-v)*p1 + u*p2 + v*p3
    # Let's return (u, v) and derive c=1-u-v if needed, or return the common (1-u-v, u, v) format
    # The original JAX returned (a, b, c) where c = 1-(a+b). Let's try to match that if possible.
    # JAX code: a = a_numerator/denominator, b = b_numerator/denominator, c = 1-(a+b)
    # Let's re-derive based on the JAX version's numerator/denominator structure.
    # It uses Cramer's rule on the system: O + t*d = p3 + a*(p1-p3) + b*(p2-p3)

    r1 = p1 - p3; r2 = p2 - p3 # Corresponds to edge2 and edge1 if p1 is origin? No, origin is p3.

    # Denominator (from JAX code structure) = Dot(d, Cross(r1, r2)) = -det from Moller-Trumbore if d is normalized? Let's stick to JAX formula.
    # Denominator = (-dx*r1y*r2z + dx*r1z*r2y + dy*r1x*r2z - dy*r1z*r2x - dz*r1x*r2y + dz*r1y*r2x)
    # This is the scalar triple product [-d, r1, r2] which is -det(d, r1, r2)
    denominator = -d[..., 0] * (r1[..., 1] * r2[..., 2] - r1[..., 2] * r2[..., 1]) \
                  -d[..., 1] * (r1[..., 2] * r2[..., 0] - r1[..., 0] * r2[..., 2]) \
                  -d[..., 2] * (r1[..., 0] * r2[..., 1] - r1[..., 1] * r2[..., 0])

    denominator_mask = (torch.abs(denominator) < eps)
    inv_denominator = 1.0 / (denominator + denominator_mask.type_as(denominator) * eps) # Avoid div by zero

    # Vector from p3 to ray origin
    O_p3 = O - p3

    # Numerator for 'a' (coefficient for p1-p3 vector)
    # Corresponds to det(O_p3, r2, d)
    a_numerator = O_p3[..., 0] * (r2[..., 1] * d[..., 2] - r2[..., 2] * d[..., 1]) \
                + O_p3[..., 1] * (r2[..., 2] * d[..., 0] - r2[..., 0] * d[..., 2]) \
                + O_p3[..., 2] * (r2[..., 0] * d[..., 1] - r2[..., 1] * d[..., 0])

    # Numerator for 'b' (coefficient for p2-p3 vector)
    # Corresponds to det(r1, O_p3, d)
    b_numerator = r1[..., 0] * (O_p3[..., 1] * d[..., 2] - O_p3[..., 2] * d[..., 1]) \
                + r1[..., 1] * (O_p3[..., 2] * d[..., 0] - O_p3[..., 0] * d[..., 2]) \
                + r1[..., 2] * (O_p3[..., 0] * d[..., 1] - O_p3[..., 1] * d[..., 0])

    a = a_numerator * inv_denominator
    b = b_numerator * inv_denominator
    c = 1.0 - (a + b)

    # Final mask: check barycentric bounds and non-parallel ray
    mask = (a >= 0) & (b >= 0) & (c >= 0) & (~denominator_mask)
    return a, b, c, mask


# --- Cell Size Constants (moved inside compute_undc_intersection or pass as args) ---
# These depend on grid_min/grid_max which are now tensors on device.
# Define them where needed or pass them.

# --- Inside Cell Mask ---
def get_inside_cell_mask(P: torch.Tensor, ooxyz: torch.Tensor,
                         half_cell_sizes: torch.Tensor,
                         get_taper_coord_func: Callable) -> torch.Tensor:
    """Checks if world point P is inside the original cell bounds (PyTorch)."""
    # half_cell_sizes: Tensor [hx, hy, hz] on the correct device
    neg_half_cell_sizes = -half_cell_sizes

    P_tapered = get_taper_coord_func(P) # Apply scene-specific taper
    P_relative = P_tapered - ooxyz # Relative to cell center in tapered space

    # Check bounds using broadcasting
    mask = (P_relative[..., 0] >= neg_half_cell_sizes[0]) & (P_relative[..., 0] < half_cell_sizes[0]) & \
           (P_relative[..., 1] >= neg_half_cell_sizes[1]) & (P_relative[..., 1] < half_cell_sizes[1]) & \
           (P_relative[..., 2] >= neg_half_cell_sizes[2]) & (P_relative[..., 2] < half_cell_sizes[2])
    return mask


# --- Ray-UNDC Intersection (PyTorch Version with Fixes) ---
def compute_undc_intersection(
    point_grid: torch.Tensor,           # The learnable offset grid (G, G, G, 3)
    cell_xyz: torch.Tensor,             # Cell indices to check (..., N_cells, 3) long
    masks: torch.Tensor,                # Validity mask for cell_xyz (..., N_cells) bool
    rays: tuple[torch.Tensor, torch.Tensor], # Origins, Directions (..., 3)
    keep_num: int,
    # Pass grid parameters and functions explicitly
    point_grid_size: int,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    point_grid_diff_lr_scale: float,
    get_taper_coord_func: Callable,      # e.g., get_taper_coord
    inverse_taper_coord_func: Callable   # e.g., inverse_taper_coord
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes ray intersections with deformed grid cell faces (PyTorch)."""

    ray_origins, ray_directions = rays
    dtype = ray_origins.dtype; device = ray_origins.device
    batch_shape = cell_xyz.shape[:-2] # Get leading batch dimensions
    num_cells = cell_xyz.shape[-2] # Number of cells per ray to check

    # --- Precompute Cell Size Constants on correct device ---
    grid_range = grid_max - grid_min
    cell_sizes = grid_range / point_grid_size
    half_cell_sizes = cell_sizes / 2.0
    cell_size_x, cell_size_y, cell_size_z = cell_sizes[0], cell_sizes[1], cell_sizes[2]
    # Create offset vectors as tensors (relative offsets in TAPERED space)
    offset_vecs = {
        "obb": torch.tensor([0,-cell_size_y,-cell_size_z], dtype=dtype, device=device), "obd": torch.tensor([0,-cell_size_y,cell_size_z], dtype=dtype, device=device),
        "odb": torch.tensor([0,cell_size_y,-cell_size_z], dtype=dtype, device=device), "odd": torch.tensor([0,cell_size_y,cell_size_z], dtype=dtype, device=device),
        "obo": torch.tensor([0,-cell_size_y,0], dtype=dtype, device=device),             "oob": torch.tensor([0,0,-cell_size_z], dtype=dtype, device=device),
        "odo": torch.tensor([0,cell_size_y,0], dtype=dtype, device=device),             "ood": torch.tensor([0,0,cell_size_z], dtype=dtype, device=device),
        "bob": torch.tensor([-cell_size_x,0,-cell_size_z], dtype=dtype, device=device), "bod": torch.tensor([-cell_size_x,0,cell_size_z], dtype=dtype, device=device),
        "dob": torch.tensor([cell_size_x,0,-cell_size_z], dtype=dtype, device=device), "dod": torch.tensor([cell_size_x,0,cell_size_z], dtype=dtype, device=device),
        "boo": torch.tensor([-cell_size_x,0,0], dtype=dtype, device=device),             "doo": torch.tensor([cell_size_x,0,0], dtype=dtype, device=device),
        "bbo": torch.tensor([-cell_size_x,-cell_size_y,0], dtype=dtype, device=device), "bdo": torch.tensor([-cell_size_x,cell_size_y,0], dtype=dtype, device=device),
        "dbo": torch.tensor([cell_size_x,-cell_size_y,0], dtype=dtype, device=device), "ddo": torch.tensor([cell_size_x,cell_size_y,0], dtype=dtype, device=device),
    }

    # --- Calculate Undeformed Cell Centers ---
    # cell_xyz expected shape (..., N_cells, 3), ensure it's float for calculation
    ooxyz = cell_xyz.type_as(grid_min) * cell_sizes + grid_min + half_cell_sizes # (..., N_cells, 3) Tapered space

    # --- Get Deformed Vertex Positions ---
    cell_x_orig, cell_y_orig, cell_z_orig = cell_xyz[..., 0], cell_xyz[..., 1], cell_xyz[..., 2]

    # --- Clamp Indices BEFORE calculating neighbors and fetching ---
    cell_x = torch.clamp(cell_x_orig, 0, point_grid_size - 1)
    cell_y = torch.clamp(cell_y_orig, 0, point_grid_size - 1)
    cell_z = torch.clamp(cell_z_orig, 0, point_grid_size - 1)
    cell_x1 = torch.clamp(cell_x + 1, 0, point_grid_size - 1)
    cell_y1 = torch.clamp(cell_y + 1, 0, point_grid_size - 1)
    cell_z1 = torch.clamp(cell_z + 1, 0, point_grid_size - 1)
    cell_x0 = torch.clamp(cell_x - 1, 0, point_grid_size - 1)
    cell_y0 = torch.clamp(cell_y - 1, 0, point_grid_size - 1)
    cell_z0 = torch.clamp(cell_z - 1, 0, point_grid_size - 1)
    # --- End Clamping ---

    # Fetch offsets from point_grid (G, G, G, 3) using CLAMPED long indices
    ooo_offset = point_grid[cell_x, cell_y, cell_z] * point_grid_diff_lr_scale
    obb_offset = point_grid[cell_x, cell_y0, cell_z0] * point_grid_diff_lr_scale
    obo_offset = point_grid[cell_x, cell_y0, cell_z] * point_grid_diff_lr_scale
    oob_offset = point_grid[cell_x, cell_y, cell_z0] * point_grid_diff_lr_scale
    odo_offset = point_grid[cell_x, cell_y1, cell_z] * point_grid_diff_lr_scale
    ood_offset = point_grid[cell_x, cell_y, cell_z1] * point_grid_diff_lr_scale
    odd_offset = point_grid[cell_x, cell_y1, cell_z1] * point_grid_diff_lr_scale
    odb_offset = point_grid[cell_x, cell_y1, cell_z0] * point_grid_diff_lr_scale
    obd_offset = point_grid[cell_x, cell_y0, cell_z1] * point_grid_diff_lr_scale
    bob_offset = point_grid[cell_x0, cell_y, cell_z0] * point_grid_diff_lr_scale
    boo_offset = point_grid[cell_x0, cell_y, cell_z] * point_grid_diff_lr_scale
    bod_offset = point_grid[cell_x0, cell_y, cell_z1] * point_grid_diff_lr_scale
    dob_offset = point_grid[cell_x1, cell_y, cell_z0] * point_grid_diff_lr_scale
    doo_offset = point_grid[cell_x1, cell_y, cell_z] * point_grid_diff_lr_scale
    dod_offset = point_grid[cell_x1, cell_y, cell_z1] * point_grid_diff_lr_scale
    bbo_offset = point_grid[cell_x0, cell_y0, cell_z] * point_grid_diff_lr_scale
    dbo_offset = point_grid[cell_x1, cell_y0, cell_z] * point_grid_diff_lr_scale
    bdo_offset = point_grid[cell_x0, cell_y1, cell_z] * point_grid_diff_lr_scale
    ddo_offset = point_grid[cell_x1, cell_y1, cell_z] * point_grid_diff_lr_scale

    # Calculate world positions of deformed vertices
    # Add offset in tapered space, then convert back to world
    ooo = inverse_taper_coord_func(ooo_offset + ooxyz)
    obb = inverse_taper_coord_func(obb_offset + ooxyz + offset_vecs["obb"])
    obo = inverse_taper_coord_func(obo_offset + ooxyz + offset_vecs["obo"])
    oob = inverse_taper_coord_func(oob_offset + ooxyz + offset_vecs["oob"])
    odo = inverse_taper_coord_func(odo_offset + ooxyz + offset_vecs["odo"])
    ood = inverse_taper_coord_func(ood_offset + ooxyz + offset_vecs["ood"])
    odd = inverse_taper_coord_func(odd_offset + ooxyz + offset_vecs["odd"])
    odb = inverse_taper_coord_func(odb_offset + ooxyz + offset_vecs["odb"])
    obd = inverse_taper_coord_func(obd_offset + ooxyz + offset_vecs["obd"])
    bob = inverse_taper_coord_func(bob_offset + ooxyz + offset_vecs["bob"])
    boo = inverse_taper_coord_func(boo_offset + ooxyz + offset_vecs["boo"])
    bod = inverse_taper_coord_func(bod_offset + ooxyz + offset_vecs["bod"])
    dob = inverse_taper_coord_func(dob_offset + ooxyz + offset_vecs["dob"])
    doo = inverse_taper_coord_func(doo_offset + ooxyz + offset_vecs["doo"])
    dod = inverse_taper_coord_func(dod_offset + ooxyz + offset_vecs["dod"])
    bbo = inverse_taper_coord_func(bbo_offset + ooxyz + offset_vecs["bbo"])
    dbo = inverse_taper_coord_func(dbo_offset + ooxyz + offset_vecs["dbo"])
    bdo = inverse_taper_coord_func(bdo_offset + ooxyz + offset_vecs["bdo"])
    ddo = inverse_taper_coord_func(ddo_offset + ooxyz + offset_vecs["ddo"])

    # Ray origin and direction, unsqueezed
    o = ray_origins.unsqueeze(-2) # (..., 1, 3)
    d = ray_directions.unsqueeze(-2) # (..., 1, 3)

    # --- Intersect Ray with 24 Triangles ---
    all_P = []
    all_mask = []
    # Define triangle vertices explicitly for clarity
    triangles = [
        (obb, obo, ooo), (obb, oob, ooo), (odd, odo, ooo), (odd, ood, ooo), # X faces (-y,-z), (-y,+z), (+y,-z), (+y,+z) ? Check convention
        (oob, odo, ooo), (oob, odo, odb), (obo, ood, ooo), (obo, ood, obd),
        (bob, boo, ooo), (bob, oob, ooo), (dod, doo, ooo), (dod, ood, ooo), # Y faces (-x,-z), (-x,+z), (+x,-z), (+x,+z)?
        (oob, doo, ooo), (oob, doo, dob), (boo, ood, ooo), (boo, ood, bod),
        (bbo, boo, ooo), (bbo, obo, ooo), (ddo, doo, ooo), (ddo, odo, ooo), # Z faces (-x,-y), (-x,+y), (+x,-y), (+x,+y)?
        (obo, doo, ooo), (obo, doo, dbo), (boo, odo, ooo), (boo, odo, bdo),
    ]

    for p1, p2, p3 in triangles:
        a, b, c, bary_mask = get_barycentric(p1, p2, p3, o, d)
        # Calculate intersection point using barycentric coordinates
        P_ = p1 * a.unsqueeze(-1) + p2 * b.unsqueeze(-1) + p3 * c.unsqueeze(-1)
        # Check if point is inside the original cell bounds
        inside_mask = get_inside_cell_mask(P_, ooxyz, half_cell_sizes, get_taper_coord_func)
        # Combine masks: must be inside cell, valid barycentric hit, and original cell mask must be true
        # Ensure masks tensor shape (..., N_cells) broadcasts correctly with inside_mask/bary_mask (..., N_cells)
        final_mask = inside_mask & bary_mask & masks
        all_P.append(P_)
        all_mask.append(final_mask)

    # Concatenate results
    world_positions = torch.stack(all_P, dim=-2)    # (..., N_cells, 24, 3)
    world_masks = torch.stack(all_mask, dim=-1)   # (..., N_cells, 24)

    # Flatten the cell and triangle dimensions for sorting
    num_total_intersections = num_cells * 24
    world_positions_flat = world_positions.reshape(*batch_shape, num_total_intersections, 3)
    world_masks_flat = world_masks.reshape(*batch_shape, num_total_intersections)

    # --- Sort and Filter Intersections ---
    # Calculate distance t = dot(P, d) - dot(O, d) (projection distance)
    # d unsqueezed: (..., 1, 3) -> broadcasts with P_flat (..., N_inter, 3)
    # o unsqueezed: (..., 1, 3)
    world_tx_flat = torch.sum((world_positions_flat - o) * d, dim=-1)

    # Mask invalid intersections (set distance to infinity)
    inf_val = 1e10
    world_tx_masked = torch.where(world_masks_flat, world_tx_flat, inf_val)

    # Sort intersections by distance
    world_tx_sorted, indices_sorted = torch.sort(world_tx_masked, dim=-1)

    # Keep the closest 'keep_num' intersections
    actual_keep_num = min(keep_num, num_total_intersections)
    top_k_tx = world_tx_sorted[..., :actual_keep_num]
    top_k_indices = indices_sorted[..., :actual_keep_num]

    # Gather the kept intersections and their properties using torch.gather
    world_masks_kept = torch.gather(world_masks_flat, -1, top_k_indices)
    top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 3) # Expand for 3D points
    world_positions_kept = torch.gather(world_positions_flat, -2, top_k_indices_expanded)

    # Convert world positions to tapered coordinates
    taper_positions_kept = get_taper_coord_func(world_positions_kept)
    # Apply mask (zero out positions where mask is false)
    taper_positions_masked = taper_positions_kept * world_masks_kept.unsqueeze(-1).type_as(taper_positions_kept)

    # Detach distances from gradient graph
    world_tx_final = top_k_tx.detach()

    # Central offset for the loss (ooo_) - detach from graph
    # Ensure shape matches batch shape if needed
    ooo_offset_detached = ooo_offset.detach()
    # Reshape regularization term to match batch shape if necessary
    # Example: if batch_shape = (B,) and ooo_offset is (B, N_cells, 3) -> take first cell? or mean?
    # Original JAX returned ooo_ * masks[..., None], masks shape (..., N_cells)
    # Let's return the detached offset for the first cell per ray for simplicity
    # OR return the mean offset for valid cells? The JAX code returned per-cell offset.
    # Let's return the offsets corresponding to the input cell_xyz, detached.
    ooo_reg_term = ooo_offset_detached # Shape (..., N_cells, 3)

    return taper_positions_masked, world_masks_kept, ooo_reg_term, world_tx_final
# (Code from previous blocks: imports, config, device setup, directory setup,
#  write_floatpoint_image, data loading, pose helpers, basic math helpers,
#  ray generation helpers, lr helpers, scene parameter setup,
#  grid parameter setup, get_acc_grid_masks)

# --- Distance Calculation Helpers (PyTorch) ---

def compute_t_forwardfacing_torch(taper_positions: torch.Tensor,
                                  world_masks: torch.Tensor,
                                  grid_max: torch.Tensor) -> torch.Tensor:
    """Calculates distance between consecutive points in tapered space (PyTorch)."""
    # Add bogus point at the end using grid_max for distance calculation
    # Shape: (..., N_pts, 3) -> (..., N_pts+1, 3)
    # Need to handle the shape carefully depending on how grid_max is provided
    # Assuming grid_max is (3,)
    end_point = grid_max.expand_as(taper_positions[..., 0:1, :]) # Shape match first point
    # Create points relative to the first point
    # Pad taper_positions relative to the first point
    relative_taper_positions = taper_positions - taper_positions[..., 0:1, :]
    # Replace masked positions with the 'end_point' relative to the first point
    masked_relative_pos = torch.where(world_masks.unsqueeze(-1),
                                      relative_taper_positions,
                                      end_point - taper_positions[..., 0:1, :])

    # Calculate squared distance from the first point
    dist_sq = torch.sum(masked_relative_pos**2, dim=-1) # Shape: (..., N_pts)
    dist = torch.sqrt(dist_sq + 1e-8) # Add epsilon for sqrt stability

    # The distortion loss expects distances *between* adjacent samples.
    # The original JAX implementation calculated `sum(w*w*delta)` and `sum(wi*wj*|ti-tj|)`.
    # The 'fake_t' returned by compute_undc_intersection seems to be distances from origin.
    # This function might need re-evaluation based on how lossfun_distortion expects input 't'.
    # Let's assume 't' should be the cumulative distance along the ray for now.
    # Note: Original JAX used sum((pos + masked*grid_max - pos[0])**2)**0.5 which is distance from origin.
    # Let's replicate that distance-from-origin calculation.
    # Using grid_max for invalid points is strange, perhaps a large value is intended? Let's use 2.0 like real360.
    # This matches the real360 structure more closely if fake_t is dist from origin.
    world_tx = torch.sqrt(torch.sum((taper_positions + (1.0 - world_masks.float().unsqueeze(-1)) * 2.0 \
                                     - taper_positions[..., 0:1, :])**2, dim=-1) + 1e-8)

    return world_tx.detach() # Detach as gradients are not needed through this

def sort_and_compute_t_real360_torch(taper_positions: torch.Tensor,
                                     world_masks: torch.Tensor
                                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts points and calculates distance from origin in tapered space (PyTorch)."""
    # Calculate squared distance from the first point/origin proxy (masked points go far away)
    # Use 2.0 as the large value for masked points, similar to JAX original
    world_tx_sq = torch.sum((taper_positions + (1.0 - world_masks.float().unsqueeze(-1)) * 2.0 \
                             - taper_positions[..., 0:1, :])**2, dim=-1)
    world_tx = torch.sqrt(world_tx_sq + 1e-8) # Add epsilon for sqrt stability

    # Sort based on these distances
    world_tx_sorted, indices_sorted = torch.sort(world_tx, dim=-1)

    # Gather points and masks based on sorted indices
    # Ensure indices have same number of dims as input for gather
    indices_expanded_masks = indices_sorted
    indices_expanded_points = indices_sorted.unsqueeze(-1).expand_as(taper_positions)

    taper_positions_sorted = torch.gather(taper_positions, -2, indices_expanded_points)
    world_masks_sorted = torch.gather(world_masks, -1, indices_expanded_masks)

    return taper_positions_sorted, world_masks_sorted, world_tx_sorted.detach()


# --- Outer Box Intersection (PyTorch) ---
def compute_box_intersection_torch(
    rays: tuple[torch.Tensor, torch.Tensor],
    point_grid_size: int,
    scene_grid_zcc: float, # Assuming this is available
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes ray intersections with the contracted outer box (PyTorch)."""
    ray_origins, ray_directions = rays
    dtype = ray_origins.dtype
    batch_shape = ray_origins.shape[:-1]
    num_batch_dims = len(batch_shape)
    epsilon = 1e-10
    inf_val = 1e10

    # Ensure origins/directions have unsqueezed dim for broadcasting with layers
    if num_batch_dims == 0: # Handle case with no batch dimension
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)
        batch_shape = (1,)
        num_batch_dims = 1

    ox, oy, oz = ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2), ray_origins.unsqueeze(-2)
    dx, dy, dz = ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2), ray_directions.unsqueeze(-2)

    dxm = (torch.abs(dx) < epsilon)
    dym = (torch.abs(dy) < epsilon)

    # Avoid division by zero
    safe_dx = dx + dxm.type_as(dx) * epsilon
    safe_dy = dy + dym.type_as(dy) * epsilon

    # Calculate layers and corresponding world distances
    num_box_layers = (point_grid_size // 2) + 1
    layers_ = torch.arange(num_box_layers, dtype=dtype, device=device) / (num_box_layers - 1) # [0, 1]
    # Apply contraction function
    layers_world = (torch.exp(layers_ * scene_grid_zcc) + scene_grid_zcc - 1.0) / scene_grid_zcc
    # Reshape for broadcasting: (1,...,1, num_box_layers)
    layers_world = layers_world.view(*([1] * num_batch_dims), num_box_layers)

    # Calculate intersection distances with +/- X/Y planes at these distances
    tx_p = safe_divide(layers_world - ox, safe_dx, epsilon)
    tx_n = safe_divide(-layers_world - ox, safe_dx, epsilon)
    ty_p = safe_divide(layers_world - oy, safe_dy, epsilon)
    ty_n = safe_divide(-layers_world - oy, safe_dy, epsilon)

    # Handle axis-aligned rays by setting t to infinity
    tx_p = torch.where(dxm.squeeze(-1), inf_val, tx_p)
    tx_n = torch.where(dxm.squeeze(-1), inf_val, tx_n)
    ty_p = torch.where(dym.squeeze(-1), inf_val, ty_p)
    ty_n = torch.where(dym.squeeze(-1), inf_val, ty_n)

    # Find furthest intersection for each axis
    tx = torch.where(tx_p > tx_n, tx_p, tx_n)
    ty = torch.where(ty_p > ty_n, ty_p, ty_n)

    # Determine which axis intersection was controlling
    tx_py = oy + safe_dy * tx # Y coord at X intersection time
    ty_px = ox + safe_dx * ty # X coord at Y intersection time
    t = torch.where(torch.abs(tx_py) < torch.abs(ty_px), tx, ty) # Choose based on smaller orthogonal coord

    # Check Z bounds
    t_pz = oz + dz * t
    world_masks = torch.abs(t_pz) < layers_world # Mask points outside Z bounds

    # Calculate world intersection points
    # Unsqueeze t to match ray shape: (..., num_box_layers, 1)
    world_positions = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * t.unsqueeze(-1)

    # Calculate tapered coordinates by scaling
    # layers_ needs broadcasting: (1,...,1, num_box_layers)
    layers_broadcast = layers_.view(*([1] * num_batch_dims), num_box_layers)
    # Add 1 to layers_ for scaling factor as in JAX code? (layers_+1)/layers * mask
    # Need to be careful with layers_world near 0, use safe divide
    taper_scales = safe_divide((layers_broadcast + 1.0), layers_world, epsilon) * world_masks.float()
    # Apply scaling: (..., num_box_layers, 3) * (..., num_box_layers, 1)
    taper_positions = world_positions * taper_scales.unsqueeze(-1)

    # Reshape back if input had no batch dim
    if batch_shape == (1,):
         taper_positions = taper_positions.squeeze(0)
         world_masks = world_masks.squeeze(0)

    return taper_positions, world_masks


# #%% --------------------------------------------------------------------------------
# # ## MLP setup <<< Block Start Marker
# #%%

num_bottleneck_features = 8

# --- PyTorch MLP Modules ---

# Initialization helper
def init_weights_glorot(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) # Glorot uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_zeros(m):
     if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight) # Zero init weight and bias
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RadianceFieldTorch(nn.Module):
    def __init__(self, out_dim: int, trunk_width: int = 384, trunk_depth: int = 8,
                 trunk_skip_length: int = 4, activation: Callable = nn.ReLU(),
                 pos_encoding_max_freq_power: int = 10, pos_encoding_include_identity: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.trunk_width = trunk_width
        self.trunk_depth = trunk_depth
        self.trunk_skip_length = trunk_skip_length
        self.activation = activation
        self.pos_encoding_max_freq_power = pos_encoding_max_freq_power
        self.pos_encoding_include_identity = pos_encoding_include_identity

        # Calculate input dimension after positional encoding
        input_dim = 3 # Base dimension for position
        encoding_dim = 3 * 2 * (pos_encoding_max_freq_power - 0) # 3 dims * 2 (sin,cos) * num_freqs
        if self.pos_encoding_include_identity:
            self.input_encoding_dim = input_dim + encoding_dim
        else:
            self.input_encoding_dim = encoding_dim

        # Create layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(self.input_encoding_dim, trunk_width))
        # Hidden layers
        for i in range(1, trunk_depth):
            # Determine input dim for layers after skip connection
            in_dim = trunk_width + self.input_encoding_dim if (i % trunk_skip_length == 0 and i > 0) else trunk_width
            layer = nn.Linear(in_dim, trunk_width)
            self.layers.append(layer)
        # Output layer
        self.output_layer = nn.Linear(trunk_width, out_dim)

        # Apply initialization
        self.layers.apply(init_weights_glorot)
        self.output_layer.apply(init_weights_glorot) # Initialize output layer too

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # Apply positional encoding
        encoded_inputs = sinusoidal_encoding(
            positions, 0, self.pos_encoding_max_freq_power, self.pos_encoding_include_identity)

        net = encoded_inputs
        skip_input = encoded_inputs # Keep original encoded input for skip connections
        for i, layer in enumerate(self.layers):
            # Concatenate skip connection *before* passing to the layer
            if (i > 0) and (i % self.trunk_skip_length == 0):
                net = torch.cat([net, skip_input], dim=-1)
            net = layer(net)
            # Apply activation after each hidden layer
            if i < len(self.layers) - 1: # Don't apply activation to the input layer's output yet
                net = self.activation(net)

        # Apply activation after input layer if depth is 1 (edge case)
        if self.trunk_depth == 1:
            net = self.activation(net)

        # Final output layer (no activation here)
        net = self.output_layer(net)
        return net


class MLPTorch(nn.Module):
    """Simple MLP for color head."""
    def __init__(self, input_dim: int, features: Sequence[int]):
        super().__init__()
        self.input_dim = input_dim
        self.features = features # e.g., [16, 16, 3]

        layers = []
        current_dim = input_dim
        for feat in features[:-1]: # Hidden layers
            layers.append(nn.Linear(current_dim, feat))
            layers.append(nn.ReLU())
            current_dim = feat
        # Output layer (no activation here)
        layers.append(nn.Linear(current_dim, features[-1]))

        self.network = nn.Sequential(*layers)

        # Apply initialization
        self.network.apply(init_weights_glorot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- Main Model combining Grids and MLPs ---
class MobileNeRFModel(nn.Module):
    # --- MODIFICATION START ---
    def __init__(self, point_grid_size: int, num_bottleneck_features: int,
                 config: dict, grid_min_tensor: torch.Tensor, grid_max_tensor: torch.Tensor): # Added grid tensors to args
    # --- MODIFICATION END ---
        super().__init__()
        self.point_grid_size = point_grid_size
        self.num_bottleneck_features = num_bottleneck_features
        self.config = config # Store config for accessing MLP params etc.

        # --- MODIFICATION START: Register grid bounds and zcc as buffers ---
        self.register_buffer('grid_min', grid_min_tensor)
        self.register_buffer('grid_max', grid_max_tensor)
        if 'scene_grid_zcc' in config and config['scene_grid_zcc'] is not None:
             # Ensure zcc is registered only if available
             self.register_buffer('scene_grid_zcc', torch.tensor(config['scene_grid_zcc'], dtype=torch.float32))
        else:
             # Handle cases where zcc might not be needed (e.g., synthetic)
             # Registering as None might cause issues with state_dict, maybe omit or use a flag
             self.scene_grid_zcc = None # Or don't register if None
             if scene_type == "real360": # Check if required for this type
                 print("Warning: scene_grid_zcc not provided in config for real360 model.")
        # --- MODIFICATION END ---

        # --- Learnable Grids ---
        self.point_grid = nn.Parameter(torch.zeros(
            (point_grid_size, point_grid_size, point_grid_size, 3), dtype=torch.float32))
        self.acc_grid = nn.Parameter(torch.zeros(
            (point_grid_size, point_grid_size, point_grid_size), dtype=torch.float32))

        # --- Instantiate MLPs ---
        # Filter config keys relevant for RadianceFieldTorch initialization
        rf_config_keys = ['trunk_width', 'trunk_depth', 'trunk_skip_length', 'pos_encoding_max_freq_power']
        rf_config = {k: config[k] for k in rf_config_keys if k in config}

        self.density_model = RadianceFieldTorch(out_dim=1, **rf_config)
        self.feature_model = RadianceFieldTorch(out_dim=num_bottleneck_features, **rf_config)

        color_mlp_input_dim = num_bottleneck_features + 3
        self.color_model = MLPTorch(
            input_dim=color_mlp_input_dim,
            features=config.get('color_mlp_features', [16, 16, 3])
        )

# --- Instantiate the Main Model (Example) ---
# This would typically happen before the training loop
model_config = { # Example config
    'trunk_width': 384, 'trunk_depth': 8, 'trunk_skip_length': 4,
    'pos_encoding_max_freq_power': 10, 'color_mlp_features': [16, 16, 3],
    'scene_grid_zcc': scene_grid_zcc if scene_type == "real360" else None,
    # Add any other config needed by MobileNeRFModel or its submodules
}
# --- MODIFICATION START: Pass grid_min and grid_max tensors ---
# grid_min and grid_max should be defined torch tensors on the correct device from Chunk 1
model = MobileNeRFModel(point_grid_size, num_bottleneck_features, model_config, grid_min, grid_max).to(device)
# --- MODIFICATION END ---
print("PyTorch model instantiated.")


# #%% --------------------------------------------------------------------------------
# # ## Main rendering functions <<< Block Start Marker
# #%%

def compute_volumetric_rendering_weights_with_alpha_torch(alpha: torch.Tensor) -> torch.Tensor:
    """Computes volume rendering weights using alpha compositing (PyTorch)."""
    # alpha: (..., num_samples)
    transmittance = 1.0 - alpha # (..., num_samples)
    # Transmittance shifted by one step (T_0 = 1)
    # Add a 1 at the beginning: (..., 1 + num_samples)
    transmittance_shifted = torch.cat([torch.ones_like(alpha[..., :1]), transmittance], dim=-1)
    # Accumulated transmittance T_i = product(1 - alpha_j) for j < i
    # Calculate cumprod on shifted, then take all but last: (..., num_samples)
    acc_transmittance = torch.cumprod(transmittance_shifted, dim=-1)[..., :-1]
    # Weight_i = T_i * alpha_i
    weights = acc_transmittance * alpha
    return weights


def render_rays_torch(
    rays: tuple[torch.Tensor, torch.Tensor],
    model: MobileNeRFModel, # Pass the nn.Module instance
    keep_num: int,
    threshold: float,
    wbgcolor: float,
    bg_color_val: float, # Pass pre-computed bg_color
    white_bkgd_flag: bool,
    # Pass scene parameters needed by geometry functions
    scene_type: str,
    point_grid_size: int,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    point_grid_diff_lr_scale: float,
    # Add other params if needed (e.g., scene_grid_zcc, taper funcs directly?)
) -> tuple: # Define precise return tuple later
    """Renders rays using the MobileNeRF model (PyTorch version)."""

    ray_origins, ray_directions = rays
    device = ray_origins.device

    # Pass necessary model parameters and scene configs to geometry functions
    # It might be cleaner if geometry functions are methods of the model or a renderer class
    grid_params = {
        "point_grid_size": point_grid_size,
        "grid_min": grid_min,
        "grid_max": grid_max,
    }
    ff_params = {}
    r360_params = {}
    if scene_type == "forwardfacing":
        ff_params = { # Assuming these are stored in model or config
            "scene_grid_taper": model.config.get('scene_grid_taper', 1.25),
            "log_z_start": model.config.get('log_z_start', math.log(25.0)),
            "log_z_end": model.config.get('log_z_end', math.log(1.0)),
            "log_z_diff": model.config.get('log_z_diff', math.log(25.0)-math.log(1.0)),
        }
        gridcell_func = gridcell_from_rays_forwardfacing
    elif scene_type == "real360":
        r360_params = {
             "scene_grid_zcc": model.config.get('scene_grid_zcc', 4.0) # Get from config
        }
        gridcell_func = gridcell_from_rays_real360
    else: # synthetic
        gridcell_func = gridcell_from_rays_synthetic

    # --- Step 1: Find grid cell intersections ---
    grid_indices, grid_masks = gridcell_func(
        rays, model.acc_grid, keep_num, threshold, **grid_params, **ff_params
    )

    # --- Step 2: Find UNDC intersections ---
    # Pass coordinate functions explicitly or access via model if stored there
    # Assume they are available globally/passed for now
    pts, undc_masks, points_reg_term, fake_t_from_undc = compute_undc_intersection(
        model.point_grid, grid_indices, grid_masks, rays, keep_num,
        point_grid_size, grid_min, grid_max, point_grid_diff_lr_scale,
        get_taper_coord, inverse_taper_coord
    )

    # --- Step 3: Handle Outer Box and Calculate Final Distances (fake_t) ---
    fake_t = fake_t_from_undc # Use distance from UNDC by default
    if scene_type == "forwardfacing":
        # Calculate final distances in tapered space for distortion loss
        fake_t = compute_t_forwardfacing_torch(pts, undc_masks, grid_max)
    elif scene_type == "real360":
        # Compute box intersections
        skybox_positions, skybox_masks = compute_box_intersection_torch(
            rays, point_grid_size, r360_params['scene_grid_zcc'], device
        )
        # Concatenate points and masks
        pts = torch.cat([pts, skybox_positions], dim=-2)
        # Ensure mask shapes match before cat (undc_masks might be bool, skybox float)
        combined_masks = torch.cat([undc_masks, skybox_masks], dim=-1)
        # Sort combined points and calculate final distances for distortion loss
        pts, combined_masks, fake_t = sort_and_compute_t_real360_torch(pts, combined_masks)
        # Update grid_masks to the combined & sorted mask for alpha calculation
        grid_masks = combined_masks # Use the sorted, combined mask hereafter


    # --- Step 4: Evaluate MLPs ---
    # Density Prediction
    mlp_alpha_raw = model.density_model(pts) # (..., N_samples, 1)
    # Apply activation and offset
    mlp_alpha = torch.sigmoid(mlp_alpha_raw.squeeze(-1) - 8.0) # (..., N_samples)
    # Apply mask (using the potentially updated combined_masks for real360)
    mlp_alpha = mlp_alpha * grid_masks.float() # Ensure mask is float

    # --- Step 5: Calculate Weights ---
    weights = compute_volumetric_rendering_weights_with_alpha_torch(mlp_alpha)
    acc = torch.sum(weights, dim=-1) # (...,) Accumulated opacity

    # --- Step 6: Binarized Alpha Visualization ---
    # Detach mlp_alpha before thresholding if using it only for viz
    with torch.no_grad():
        alpha_thresh = (mlp_alpha > 0.5).float()
        # Clip alpha_b to avoid exactly 0 or 1 for stability if needed by loss (not here)
        # mlp_alpha_b = torch.clamp(alpha_thresh, 1e-5, 1.0 - 1e-5)
        mlp_alpha_b = alpha_thresh # Simpler thresholding for viz
    weights_b = compute_volumetric_rendering_weights_with_alpha_torch(mlp_alpha_b)
    acc_b = torch.sum(weights_b, dim=-1)

    # --- Step 7: Evaluate Feature and Color MLPs ---
    # Normalize view directions
    viewdirs = _normalize(ray_directions) # (...)
    # Broadcast view directions to match points: (..., 1, 3) -> (..., N_samples, 3)
    dirs_expanded = viewdirs.unsqueeze(-2).expand_as(pts)

    # Feature Prediction
    mlp_features_raw = model.feature_model(pts) # (..., N_samples, feat_dim)
    mlp_features = torch.sigmoid(mlp_features_raw) # Apply activation

    # Color Prediction
    # Concatenate features and view directions
    features_dirs_enc = torch.cat([mlp_features, dirs_expanded], dim=-1)
    colors_raw = model.color_model(features_dirs_enc) # (..., N_samples, 3)
    colors = torch.sigmoid(colors_raw) # Apply activation

    # --- Step 8: Composite Final Colors ---
    # Weighted sum: (..., 1, N_samples) @ (..., N_samples, 3) -> (..., 1, 3) -> (..., 3)
    # Use unsqueeze/sum: (..., N_samples, 1) * (..., N_samples, 3) -> sum -> (..., 3)
    rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
    rgb_b = torch.sum(weights_b.unsqueeze(-1) * colors, dim=-2) # For visualization

    # --- Step 9: Composite onto Background ---
    # Ensure accumulated opacity has channel dim: (..., 1)
    acc = acc.unsqueeze(-1)
    acc_b = acc_b.unsqueeze(-1)

    if white_bkgd_flag:
        rgb = rgb + (1.0 - acc) # Add white background
        rgb_b = rgb_b + (1.0 - acc_b)
    else:
        # Use pre-calculated bg_color_val, handle wbgcolor mixing
        # For simplicity here, just use bg_color_val (wbgcolor=0)
        # bgc = torch.rand_like(acc) < wbgcolor # Random mask based on wbgcolor
        # bg_val = torch.where(bgc, 0.0, bg_color_val) # Mix black and average color
        bg_val = bg_color_val # Use average color directly
        rgb = rgb + (1.0 - acc) * bg_val
        rgb_b = rgb_b + (1.0 - acc_b) * bg_val

    # --- Step 10: Get Occupancy Masks for Loss ---
    # Need to call get_acc_grid_masks again with the final set of points 'pts'
    # (which might include skybox points for real360)
    # Assuming get_taper_coord works for the combined points
    taper_pts_final = get_taper_coord(pts)
    acc_grid_loss_masks = get_acc_grid_masks(
        taper_pts_final, model.acc_grid, grid_min, grid_max, point_grid_size
    )
    # Apply the final validity mask (grid_masks might be the combined one for real360)
    acc_grid_loss_masks = acc_grid_loss_masks * grid_masks.float()

    # --- Targeted Debug Prints ---
    print(f"[Debug render] scene_type={scene_type}")
    print(f"  grid_masks  true count={grid_masks.float().sum().item()}/{grid_masks.numel()}")
    print(f"  undc_masks  true count={undc_masks.float().sum().item()}/{undc_masks.numel()}")
    if scene_type == "real360":
        print(f"  skybox_masks true count={skybox_masks.float().sum().item()}/{skybox_masks.numel()}")
    print(f"  pts         min/max/mean = {pts.min().item():.3f}/{pts.max().item():.3f}/{pts.mean().item():.3f}")
    print(f"  mlp_alpha_raw min/max/mean = {mlp_alpha_raw.min().item():.3f}/{mlp_alpha_raw.max().item():.3f}/{mlp_alpha_raw.mean().item():.3f}")
    print(f"  mlp_alpha     min/max/mean = {mlp_alpha.min().item():.3f}/{mlp_alpha.max().item():.3f}/{mlp_alpha.mean().item():.3f}")
    per_ray = weights.sum(dim=-1)
    print(f"  weights sum/ray   min/max/mean = {per_ray.min().item():.3f}/{per_ray.max().item():.3f}/{per_ray.mean().item():.3f}")

    # Return all necessary outputs
    return rgb, acc.squeeze(-1), rgb_b, acc_b.squeeze(-1), \
           mlp_alpha, weights, points_reg_term, fake_t, acc_grid_loss_masks

# --- Evaluation Setup (PyTorch) ---
test_batch_size = 4096 # No n_device division for single GPU / handled by DDP sampler later
test_keep_num = point_grid_size * 3 // 4
test_threshold = 0.1
test_wbgcolor = 0.0 # Will use bg_color directly in render_rays_torch

# Simplified render_loop returning TENSORS on CPU
def render_loop_torch(rays: tuple[torch.Tensor, torch.Tensor],
                      model: MobileNeRFModel,
                      chunk: int,
                      # Pass necessary scene/render params
                      keep_num: int, threshold: float, bg_color_val: float, white_bkgd_flag: bool,
                      scene_type: str, point_grid_size: int, grid_min: torch.Tensor,
                      grid_max: torch.Tensor, point_grid_diff_lr_scale: float,
                      # Pass necessary taper funcs, zcc etc. if needed by render_rays_torch
                     ) -> list[torch.Tensor]: # Returns list of TENSORS on CPU

    model.eval() # Set model to evaluation mode
    origins, dirs = rays
    sh = list(origins.shape[:-1]) # e.g., [H, W]
    origins_flat = origins.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    l = origins_flat.shape[0]
    results = [] # To store chunk results (tensors on GPU initially)

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for i in tqdm(range(0, l, chunk), desc="Rendering Chunks", leave=False, disable= l < chunk*2):
            chunk_origins = origins_flat[i:i+chunk]
            chunk_dirs = dirs_flat[i:i+chunk]
            # Ensure chunk is not empty before proceeding
            if chunk_origins.shape[0] == 0:
                continue
            chunk_rays = (chunk_origins, chunk_dirs)

            # Call the main rendering function directly for the chunk
            render_outputs = render_rays_torch(
                chunk_rays, model, keep_num, threshold, 0.0, # wbgcolor=0 for test
                bg_color_val, white_bkgd_flag, scene_type, point_grid_size,
                grid_min, grid_max, point_grid_diff_lr_scale
                # Pass other required params
            )
            # Append relevant outputs (e.g., rgb, acc, rgb_b, acc_b)
            # Keep tensors on GPU within the loop
            if not results:
                 # Initialize list with empty lists matching output structure
                 results = [[] for _ in range(4)] # Assuming we want rgb, acc, rgb_b, acc_b

            results[0].append(render_outputs[0]) # rgb (GPU)
            results[1].append(render_outputs[1]) # acc (GPU)
            results[2].append(render_outputs[2]) # rgb_b (GPU)
            results[3].append(render_outputs[3]) # acc_b (GPU)

    # Concatenate chunk results on GPU first
    full_results_tensors_gpu = [torch.cat(r, dim=0) for r in results]

    # Reshape and move final outputs to CPU
    outs_tensors_cpu = []
    # Expected return: rgb, acc, rgb_b, acc_b
    for i in range(len(full_results_tensors_gpu)):
        tensor_data = full_results_tensors_gpu[i]
        # Determine expected shape
        if tensor_data.ndim > 1 and tensor_data.shape[-1] == 3: # Color image
             final_shape = sh + [3]
        else: # Grayscale/acc
             final_shape = sh
        # Reshape and move to CPU
        outs_tensors_cpu.append(tensor_data.reshape(final_shape).cpu()) # <<< MOVE TO CPU HERE

    model.train() # Set model back to training mode
    return outs_tensors_cpu # List of Tensors on CPU
# --- Initial Test Render (PyTorch) ---
model_config = { # Populate with actual desired values
    'trunk_width': 384, 'trunk_depth': 8, 'trunk_skip_length': 4,
    'pos_encoding_max_freq_power': 10, 'color_mlp_features': [16, 16, 3],
    'scene_grid_zcc': scene_grid_zcc if scene_type == "real360" else None,
    # Add any other config needed by MobileNeRFModel or its submodules
}
model = MobileNeRFModel(point_grid_size,
                       num_bottleneck_features,
                       model_config,
                       grid_min,
                       grid_max).to(device)
print("Performing initial test render...")
test_idx = 0 # Example index
test_pose = data['test']['c2w'][test_idx:test_idx+1].squeeze(0) # Get single pose (3,4)
test_hwf = data['test']['hwf']
test_gt = data['test']['images'][test_idx]

initial_rays = camera_ray_batch(test_pose, test_hwf) # Generate rays on device

# Render using the PyTorch loop
out = render_loop_torch(initial_rays, model, test_batch_size,
                        test_keep_num, test_threshold, bg_color, white_bkgd,
                        scene_type, point_grid_size, grid_min, grid_max,
                        point_grid_diff_lr_scale)

# out is now a list of NumPy arrays
rgb = out[0]; acc = out[1]; rgb_b = out[2]; acc_b = out[3]
# Save using the NumPy-based write_floatpoint_image
write_floatpoint_image(os.path.join(samples_dir, "s1_0_rgb.png"), rgb)
write_floatpoint_image(os.path.join(samples_dir, "s1_0_rgb_binarized.png"), rgb_b)
write_floatpoint_image(os.path.join(samples_dir, "s1_0_gt.png"), test_gt.cpu().numpy()) # GT is tensor
write_floatpoint_image(os.path.join(samples_dir, "s1_0_acc.png"), acc[..., None]) # Add channel dim
write_floatpoint_image(os.path.join(samples_dir, "s1_0_acc_binarized.png"), acc_b[..., None]) # Add channel dim
print("Initial test render saved.")


# --- Loss Functions (PyTorch) ---
def lossfun_distortion_torch(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Distortion loss (PyTorch version)."""
    # t: distances along ray (..., N_samples)
    # w: weights (..., N_samples)
    # Midpoints of intervals
    t_mid = 0.5 * (t[..., 1:] + t[..., :-1])
    # Lengths of intervals
    t_delta = t[..., 1:] - t[..., :-1] # (..., N_samples-1)

    # Loss_intra: sum(w_i^2 * delta_i / 3) for intervals i
    loss_intra = torch.sum(w[..., 1:]**2 * t_delta, dim=-1) / 3.0 # Not exactly matching JAX version?
    # Let's re-implement loss_self from JAX version
    loss_self = torch.sum((w[..., 1:]**2 + w[..., :-1]**2) * t_delta, dim=-1) / 6.0

    # Loss_inter: sum(w_i * w_j * |t_mid_i - t_mid_j|) for i < j
    # Calculate pairwise differences of midpoints
    # Shape: (..., N_samples-1, N_samples-1)
    dt_mid = torch.abs(t_mid.unsqueeze(-1) - t_mid.unsqueeze(-2))
    # Calculate pairwise products of weights
    # Shape: (..., N_samples-1, N_samples-1)
    weights_pairs = w[..., 1:].unsqueeze(-1) * w[..., 1:].unsqueeze(-2)
    # Sum weighted distances
    loss_cross = torch.sum(weights_pairs * dt_mid, dim=(-1, -2)) # Check dims carefully

    # Original JAX: losses_cross = np.sum(w * np.sum(w[..., None, :] * dux, axis=-1), axis=-1)
    # where dux = |t[:, None] - t[None, :]|. This used original t, not midpoints. Let's match this.
    dt_orig = torch.abs(t.unsqueeze(-1) - t.unsqueeze(-2)) # (..., N_samples, N_samples)
    weights_pairs_orig = w.unsqueeze(-1) * w.unsqueeze(-2) # (..., N_samples, N_samples)
    loss_cross_jax = torch.sum(weights_pairs_orig * dt_orig, dim=(-1, -2)) / 2.0 # Divide by 2? JAX sum was over all pairs? Check JAX code again.
    # JAX: np.sum(w * np.sum(w[..., None, :] * dux, axis=-1), axis=-1)
    inner_sum = torch.sum(w.unsqueeze(-2) * dt_orig, dim=-1) # (..., N_samples)
    loss_cross_jax_v2 = torch.sum(w * inner_sum, dim=-1) # (...)

    # Use the version matching JAX logic
    return loss_cross_jax_v2 + loss_self

def compute_TV_torch(acc_grid: torch.Tensor) -> torch.Tensor:
    """Total Variation loss for acc_grid (PyTorch)."""
    # acc_grid assumed shape (G, G, G)
    dx = acc_grid[:-1, :, :] - acc_grid[1:, :, :]
    dy = acc_grid[:, :-1, :] - acc_grid[:, 1:, :]
    dz = acc_grid[:, :, :-1] - acc_grid[:, :, 1:]
    # Use torch.mean(torch.square(...))
    TV = torch.mean(torch.square(dx)) + torch.mean(torch.square(dy)) + torch.mean(torch.square(dz))
    return TV

# --- Training Step Function (PyTorch) ---
def train_step_torch(
    model: MobileNeRFModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # Optional LR scheduler
    train_data_batch: dict, # Contains rays, pixels for the batch
    hparams: dict # Dictionary containing hyperparameters like weights, keep_num etc.
) -> dict: # Return dictionary of loss values
    """Performs one training step (PyTorch)."""
    model.train() # Set model to training mode

    rays, pixels = train_data_batch['rays'], train_data_batch['pixels']
    ray_origins, ray_dirs = rays

    # --- Forward Pass ---
    # Call render_rays_torch, passing model and necessary hparams/scene params
    # Need to make sure all required params are in hparams or model attributes
    render_outputs = render_rays_torch(
        rays=(ray_origins, ray_dirs),
        model=model,
        keep_num=hparams['keep_num'],
        threshold=hparams['threshold'],
        wbgcolor=hparams['wbgcolor'], # This might not be needed if handled differently
        bg_color_val=hparams['bg_color_val'], # Pass actual bg color
        white_bkgd_flag=hparams['white_bkgd'],
        # Pass scene params explicitly
        scene_type=hparams['scene_type'],
        point_grid_size=model.point_grid_size, # Access attribute - OK
        grid_min=model.grid_min,               # <<< CORRECT way to access registered buffer
        grid_max=model.grid_max,               # <<< CORRECT way to access registered buffer
        point_grid_diff_lr_scale=hparams['point_grid_diff_lr_scale'],
        # Pass other params if needed
    )

    # Unpack outputs needed for loss
    rgb_est, _, _, _, _, weights, points_reg, fake_t, acc_grid_loss_masks = render_outputs
    # Expected shapes: rgb_est(B,3), weights(B,Ns), points_reg(B,Nk,3), fake_t(B,Ns), acc_grid_loss_masks(B,Ns)

    # --- Calculate Loss Components ---
    # 1. Color Loss (MSE)
    loss_color_l2 = torch.nn.functional.mse_loss(rgb_est, pixels)

    # 2. Accuracy Grid Loss
    # Ensure weights are detached if gradient shouldn't flow through them here
    loss_acc = torch.mean(torch.clamp(weights.detach() - acc_grid_loss_masks, min=0))
    # Regularization on acc_grid (Parameter inside model)
    loss_acc += torch.mean(torch.abs(model.acc_grid)) * 1e-5
    loss_acc += compute_TV_torch(model.acc_grid) * 1e-5

    # 3. Distortion Loss
    loss_distortion = torch.mean(lossfun_distortion_torch(fake_t, weights)) * hparams['wdistortion']

    # 4. Point Grid Regularization Loss
    point_loss_abs = torch.abs(points_reg) # points_reg_term is ooo_offset from UNDC
    # Calculate cell size (needs grid_min/max/size)
    grid_range = model.grid_max - model.grid_min
    half_cell_size = (grid_range / model.point_grid_size / 2.0).norm() # Use norm as threshold? Or check per dim?
    # Let's check per dimension mean absolute value
    half_cell_dims = grid_range / model.point_grid_size / 2.0

    point_loss_out = point_loss_abs * 1000.0
    point_loss_in = point_loss_abs * 0.01
    # Check if absolute offset exceeds half cell size along any dimension
    point_mask = torch.any(point_loss_abs > half_cell_dims, dim=-1, keepdim=True)
    point_loss = torch.mean(torch.where(point_mask, point_loss_out, point_loss_in))

    # --- Total Loss ---
    total_loss = loss_color_l2 + loss_distortion + loss_acc + point_loss

    # --- Backward Pass and Optimization ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Return scalar loss values for logging
    return {
        'total_loss': total_loss.item(),
        'loss_color_l2': loss_color_l2.item(),
        'loss_acc': loss_acc.item(),
        'loss_distortion': loss_distortion.item(),
        'point_loss': point_loss.item(),
    }


# --- Training Loop Setup (PyTorch) ---
# Instantiate model, optimizer, scheduler
model = MobileNeRFModel(point_grid_size,
                       num_bottleneck_features,
                       model_config,
                       grid_min,
                       grid_max).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, # Initial LR (will be adjusted)
                             betas=(adam_kwargs['beta1'], adam_kwargs['beta2']),
                             eps=adam_kwargs['eps'])
# Example using LambdaLR with the converted lr_fn
# Define max steps for schedule
train_iters_cont = 300000 if scene_type == "real360" else 200000 # Max steps for LR schedule
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: lr_fn(step, train_iters_cont, 1.0, 0.1) # lr0=1, lr1=0.1 relative factors
    # Note: Actual initial LR is set in Adam. This lambda returns a multiplier. Adjust lr0/lr1 if lr_fn expects absolute values.
    # Let's assume lr_fn returns absolute values for now.
    # lr_lambda=lambda step: lr_fn(step, train_iters_cont, 1e-3, 1e-5) / optimizer.defaults['lr'] # Relative
)

# Load checkpoint if resuming
step_init = 0
# checkpoint_path = "path/to/checkpoint.pth"
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     step_init = checkpoint['step']
#     lr_scheduler.last_epoch = step_init # Important for scheduler resumption
#     print(f"Resumed from step {step_init}")

print(f'Starting at step {step_init}')

# Training loop variables
psnrs = [] # List to store training PSNR
iters = [] # Iteration numbers for psnrs plot
psnrs_test = [] # Test PSNR periodically
iters_test = [] # Iteration numbers for psnrs_test plot
t_total = 0.0; t_last = 0.0; i_last = step_init

# Total training iterations
training_iters = 200000 if scene_type != "real360" else 300000


# --- Main Training Loop (PyTorch) ---
print("Training")
# Use range starting from step_init
for i in tqdm(range(step_init, training_iters + 1), initial=step_init, total=training_iters+1):
    t_start = time.time()

    # --- Dynamic Hyperparameter Calculation ---
    # LR is handled by scheduler.step() after optimizer.step()
    current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging if needed

    wbgcolor = min(1.0, float(i) / 50000) # Original logic for background mixing (might adapt)
    wbinary = 0.0 # Not used in stage 1 loss

    if scene_type == "synthetic": wdistortion = 0.0
    elif scene_type == "forwardfacing": wdistortion = 0.0 if i < 10000 else 0.01
    elif scene_type == "real360": wdistortion = 0.0 if i < 10000 else 0.001

    # Coarse-to-fine sampling strategy
    if i <= 50000:
        batch_size = 1024 # Use eval batch size as base
        keep_num = test_keep_num * 4
        threshold = -100000.0
    elif i <= 100000:
        batch_size = 1024
        keep_num = test_keep_num * 2
        threshold = test_threshold
    else:
        batch_size = 1024
        keep_num = test_keep_num
        threshold = test_threshold

    # --- Get Data Batch ---
    # Assuming random_ray_batch is adapted for PyTorch and takes device
    batch_rays, batch_pixels = random_ray_batch(batch_size, data['train'], device)

    # --- Execute Training Step ---
    hparams = {
        'keep_num': keep_num, 'threshold': threshold, 'wbgcolor': wbgcolor,
        'wdistortion': wdistortion, 'white_bkgd': white_bkgd, 'bg_color_val': bg_color,
        'scene_type': scene_type, 'point_grid_diff_lr_scale': point_grid_diff_lr_scale,
    }
    loss_dict = train_step_torch(model, optimizer, lr_scheduler,
                                 {'rays': batch_rays, 'pixels': batch_pixels},
                                 hparams)

    # --- Record Metrics ---
    # Calculate PSNR from L2 loss
    if loss_dict['loss_color_l2'] > 0:
        psnr = -10.0 * math.log10(loss_dict['loss_color_l2'])
    else:
        psnr = float('inf') # Handle perfect score case
    psnrs.append(psnr)
    iters.append(i)

    t_end = time.time()
    t_total += (t_end - t_start)

    # --- Periodic Logging & Evaluation ---
    if (i % 10000 == 0) and i >= 0: # Log at step 0 and every 10k
        gc.collect(); torch.cuda.empty_cache() # Clear memory

        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(weights_dir, f"s1_tmp_state{i}.pth")
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        }, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path}")

        # --- Print Training Stats ---
        avg_psnr_train = numpy.mean(psnrs[-200:]) # Avg over last 200 iters
        print(f'Iter: {i}, Elapsed: {t_total//60:.0f}m {t_total%60:.0f}s, LR: {current_lr:.2e}')
        print(f'  Loss: {loss_dict["total_loss"]:.4f}, PSNR (train avg): {avg_psnr_train:.3f}')
        print(f'  Batch Size: {batch_size}, Keep Num: {keep_num}')
        if i > i_last: # Avoid division by zero at step 0
             t_elapsed = t_total - t_last; i_elapsed = i - i_last
             print(f"  Speed: {t_elapsed / i_elapsed:.3f} sec/iter, {i_elapsed / t_elapsed:.3f} iter/sec")
        t_last = t_total; i_last = i

        # --- Evaluate on Selected Test Image ---
        print("  Evaluating test image...")
        test_idx = 0 # Choose appropriate index based on scene_type if needed
        test_pose = data['test']['c2w'][test_idx:test_idx+1].squeeze(0)
        test_hwf = data['test']['hwf']
        test_gt = data['test']['images'][test_idx]
        eval_rays = camera_ray_batch(test_pose, test_hwf)

        # Render using the PyTorch loop
        eval_out = render_loop_torch(
            eval_rays, model, test_batch_size, # Use test batch size for eval chunk
            test_keep_num, test_threshold, bg_color, white_bkgd,
            scene_type, point_grid_size, grid_min, grid_max,
            point_grid_diff_lr_scale
        )
        rgb_np = eval_out[0]; acc_np = eval_out[1]
        rgb_b_np = eval_out[2]; acc_b_np = eval_out[3]

        # Debug prints: check range and mean of your renders
        if isinstance(rgb_np, torch.Tensor):
            rgb_arr = rgb_np.detach().cpu().numpy()
        else:
            rgb_arr = rgb_np
        if isinstance(rgb_b_np, torch.Tensor):
            rgbb_arr = rgb_b_np.detach().cpu().numpy()
        else:
            rgbb_arr = rgb_b_np
        print(f"  Debug RGB  — mean/min/max: {rgb_arr.mean():.4f}/{rgb_arr.min():.4f}/{rgb_arr.max():.4f}")
        print(f"  Debug ACC  — mean/min/max: {acc_np.mean():.4f}/{acc_np.min():.4f}/{acc_np.max():.4f}")
        print(f"  Debug RGBb — mean/min/max: {rgbb_arr.mean():.4f}/{rgbb_arr.min():.4f}/{rgbb_arr.max():.4f}")
        print(f"  Debug ACCb — mean/min/max: {acc_b_np.mean():.4f}/{acc_b_np.min():.4f}/{acc_b_np.max():.4f}")

        # Calculate PSNR for this test image
        # Ensure both are numpy arrays before calculation
        if isinstance(rgb_np, torch.Tensor):
            rgb_np_array = rgb_np.detach().cpu().numpy()
        else:
            rgb_np_array = rgb_np
        test_gt_array = test_gt.detach().cpu().numpy()
        
        test_mse = numpy.mean(numpy.square(rgb_np_array - test_gt_array))
        test_psnr = -10.0 * numpy.log10(test_mse) if test_mse > 0 else float('inf')
        psnrs_test.append(test_psnr)
        iters_test.append(i)
        print(f'  PSNR (test selected): {test_psnr:.3f}')

        # --- Plot Loss Curve ---
        plt.figure(); plt.title(f"Stage 1 Training @ Step {i}")
        plt.plot(iters, psnrs, label="Train PSNR (batch)")
        plt.plot(iters_test, psnrs_test, label="Test PSNR (selected)", marker='o')
        p = numpy.array(psnrs_test + psnrs) # Combine for ylim calculation
        plt.ylim(numpy.nanmin(p) - 1.0, numpy.nanmax(p) + 1.0) # Use nanmin/max
        plt.xlabel("Iteration"); plt.ylabel("PSNR (dB)")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(samples_dir, f"s1_{i}_loss.png"))
        plt.close()

        # --- Save Visualization Images ---
        write_floatpoint_image(os.path.join(samples_dir, f"s1_{i}_rgb.png"), rgb_np)
        write_floatpoint_image(os.path.join(samples_dir, f"s1_{i}_rgb_binarized.png"), rgb_b_np)
        write_floatpoint_image(os.path.join(samples_dir, f"s1_{i}_gt.png"), test_gt) # Save original tensor
        write_floatpoint_image(os.path.join(samples_dir, f"s1_{i}_acc.png"), acc_np[..., None])
        write_floatpoint_image(os.path.join(samples_dir, f"s1_{i}_acc_binarized.png"), acc_b_np[..., None])
        print("  Logging complete.")


# #%% --------------------------------------------------------------------------------
# # ## Run test-set evaluation <<< Block Start Marker
# #%%
print("\nRunning final test set evaluation...")
gc.collect(); torch.cuda.empty_cache()

# Load final/best model checkpoint
# Assuming the last saved checkpoint is the final one for simplicity
final_checkpoint_path = os.path.join(weights_dir, f"s1_tmp_state{training_iters}.pth")
if os.path.exists(final_checkpoint_path):
    checkpoint = torch.load(final_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded final model state from {final_checkpoint_path}")
else:
    print("Warning: Final checkpoint not found. Evaluating with current model state.")

model.eval() # Set model to evaluation mode

# Get all test poses and images
test_poses = data['test']['c2w']
test_images = data['test']['images']
test_hwf = data['test']['hwf']
num_test_images = test_poses.shape[0]

all_psnrs = []
all_ssims = []
render_frames = [] # Optional: store rendered frames

# Import SSIM function
from pytorch_msssim import ssim

with torch.no_grad(): # Ensure no gradients are computed
    for i in tqdm(range(num_test_images), desc="Evaluating Test Set"):
        pose = test_poses[i]
        gt_image = test_images[i]
        rays = camera_ray_batch(pose, test_hwf)

        # Render the image
        render_out = render_loop_torch(
            rays, model, test_batch_size,
            test_keep_num, test_threshold, bg_color, white_bkgd,
            scene_type, point_grid_size, grid_min, grid_max,
            point_grid_diff_lr_scale
        )
        rgb_np = render_out[0]
        # Optional: store frame
        # render_frames.append(rgb_np)

        # Calculate PSNR
        gt_np = gt_image.cpu().numpy()
        mse = numpy.mean(numpy.square(rgb_np - gt_np))
        psnr = -10.0 * numpy.log10(mse) if mse > 0 else float('inf')
        all_psnrs.append(psnr)

        # Calculate SSIM
        # Ensure tensors are (N, C, H, W) and float
        img0_torch = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img1_torch = gt_image.permute(2, 0, 1).unsqueeze(0).float().to(device)
        ssim_val = ssim(img0_torch, img1_torch, data_range=1.0, size_average=True).item()
        all_ssims.append(ssim_val)

# Calculate and print average metrics
avg_psnr = numpy.mean(all_psnrs)
avg_ssim = numpy.mean(all_ssims)
print(f"\nTest Set Evaluation Complete:")
print(f"  Average PSNR: {avg_psnr:.4f}")
print(f"  Average SSIM: {avg_ssim:.4f}")

# #%% --------------------------------------------------------------------------------
# # ## Save weights <<< Block Start Marker
# #%%
final_weights_path = os.path.join(weights_dir, "weights_stage1.pth") # Use .pth extension
print(f"\nSaving final model state dictionary to {final_weights_path}")
# Save only the model state_dict
torch.save(model.state_dict(), final_weights_path)
print("Final weights saved.")