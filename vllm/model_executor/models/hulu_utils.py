import os
import warnings
from typing import List, Optional, Union
import numpy as np
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """
    Load a single image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object in RGB format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_path}: {e}")


def load_images(image_path: Union[str, List[str], Image.Image, List[Image.Image]]) -> List[Image.Image]:
    """
    Load multiple images. Matches HulumedProcessor.load_images() logic.
    
    Args:
        image_path: Single file path, list of paths, directory path, PIL Image, or list of PIL Images
        
    Returns:
        List of PIL Images in RGB format
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If no valid images found
    """
    if isinstance(image_path, str) and os.path.isfile(image_path):
        # Single file path
        images = [Image.open(image_path).convert('RGB')]
    elif isinstance(image_path, str) and os.path.isdir(image_path):
        # Directory path
        image_files = sorted([
            f for f in os.listdir(image_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))
        ])
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_path}")
        images = [Image.open(os.path.join(image_path, f)).convert('RGB') for f in image_files]
    elif isinstance(image_path, list) and len(image_path) > 0 and isinstance(image_path[0], str):
        # List of file paths
        images = [Image.open(f).convert('RGB') for f in image_path]
    elif isinstance(image_path, list) and len(image_path) > 0 and isinstance(image_path[0], Image.Image):
        # List of PIL Images - convert to RGB if needed
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in image_path]
    elif isinstance(image_path, Image.Image):
        # Single PIL Image
        images = [image_path.convert('RGB') if image_path.mode != 'RGB' else image_path]
    else:
        raise ValueError(f"Unsupported image path type: {type(image_path)}")
    
    return images


def load_video(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 16
) -> List[Image.Image]:
    """
    Load video and extract frames using FFmpeg.
    
    Args:
        video_path: Path to video file
        fps: Target frame rate for sampling
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of PIL Image objects
        
    Raises:
        ImportError: If ffmpeg-python is not installed
        FileNotFoundError: If video file doesn't exist
    """
    try:
        import ffmpeg
    except ImportError:
        raise ImportError(
            "ffmpeg-python is required for video processing.\n"
            "Install it with: pip install ffmpeg-python"
        )
    
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        # Probe video
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        w, h = int(video_stream['width']), int(video_stream['height'])
        
        # Extract frames
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='down')
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
        
        out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
        frames_np = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        
        # Subsample if needed
        if len(frames_np) > max_frames:
            indices = np.linspace(0, len(frames_np) - 1, max_frames, dtype=int)
            frames_np = frames_np[indices]
        
        frames = [Image.fromarray(frame) for frame in frames_np]
        return frames
        
    except Exception as e:
        raise RuntimeError(f"Failed to process video {video_path}: {e}")


def load_3d(
    nii_path: str,
    num_slices: Optional[int] = None,
    axis: int = 2
) -> List[Image.Image]:
    """
    Load 3D medical volume (NIfTI) and extract 2D slices.
    
    Args:
        nii_path: Path to NIfTI file (.nii or .nii.gz)
        num_slices: Number of slices to extract (uniformly sampled). If None, extract all slices
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        List of PIL Image objects (grayscale converted to RGB)
        
    Raises:
        ImportError: If nibabel is not installed
        FileNotFoundError: If NIfTI file doesn't exist
        ValueError: If axis is invalid
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI support.\n"
            "Install it with: pip install nibabel"
        )
    
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"NIfTI file not found: {nii_path}")
    
    # Load NIfTI
    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata()
    
    # Extract slices along specified axis
    if axis == 0:
        slices = [volume[i, :, :] for i in range(volume.shape[0])]
    elif axis == 1:
        slices = [volume[:, i, :] for i in range(volume.shape[1])]
    elif axis == 2:
        slices = [volume[:, :, i] for i in range(volume.shape[2])]
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")
    
    # Subsample slices if needed
    if num_slices is not None and num_slices < len(slices):
        indices = np.linspace(0, len(slices) - 1, num_slices, dtype=int)
        slices = [slices[i] for i in indices]
    
    # Process each slice
    processed_slices = []
    for slice_2d in slices:
        # Normalize to 0-255
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            slice_2d = (slice_2d - slice_min) / (slice_max - slice_min) * 255.0
        else:
            slice_2d = np.zeros_like(slice_2d)
        
        slice_2d = slice_2d.astype(np.uint8)
        
        # Convert grayscale to RGB
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        processed_slices.append(Image.fromarray(slice_rgb))
    
    return processed_slices
