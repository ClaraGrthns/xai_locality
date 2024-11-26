import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    
    return transf

def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]) 
    ])    
    return transf 

def grid_segments(image, n_segments = 150):
    """Creates a grid segmentation of an image.
    
    Args:
        image: numpy array of shape [H, W, C] or [H, W]
        n_segments: int, number of grid segments (will be squared to get total segments)
    
    Returns:
        segments: numpy array of shape [H, W] where each unique value represents a segment
    """

    if len(image.shape) == 3 and type(image)==np.ndarray:
        height, width, _ = image.shape
    elif len(image.shape) == 3 and type(image)==torch.Tensor:
        _, height, width = image.shape
        image = image.numpy()
    else:
        height, width = image.shape
        
    # Calculate grid size
    n_segments_side = int(np.sqrt(n_segments))
    h_step = height // n_segments_side
    w_step = width // n_segments_side
    
    # Create segment labels
    segments = np.zeros((height, width), dtype=np.int64)
    for i in range(n_segments_side):
        for j in range(n_segments_side):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < n_segments_side - 1 else height
            w_start = j * w_step
            w_end = (j + 1) * w_step if j < n_segments_side - 1 else width
            segments[h_start:h_end, w_start:w_end] = i * n_segments_side + j
            
    return segments

def grid_wrapper(n_segments):
    def wrapper(image):
        return grid_segments(image, n_segments)
    return wrapper

