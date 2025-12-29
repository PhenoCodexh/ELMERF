import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
from mmseg.utils import get_palette

# Path to the configuration file
config_file = r'path/to/your/config.py'
# Path to the model checkpoint file
checkpoint_file = r"path/to/your/checkpoint.pth"

# Initialize the model for inference on GPU
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Path to the input images folder
img_root = Path("path/to/input/images")
# Path to save the segmentation masks
save_mask_root = Path("path/to/output/masks")
# Create the output directory if it doesn't exist
save_mask_root.mkdir(parents=True, exist_ok=True)

# Iterate through all images in the input folder
for img_path in tqdm(list(img_root.iterdir())):
    # Filter out non-image files
    if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
        continue

    # Perform inference on a single image
    data_sample = inference_model(model, str(img_path))

    # Compatible with both old and new MMSegmentation output formats
    if hasattr(data_sample, 'pred_sem_seg'):
        # New version: extract segmentation result from pred_sem_seg attribute
        seg = data_sample.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    else:
        # Old version: directly get the segmentation result
        seg = data_sample[0].astype(np.uint8)

    # Convert segmentation indices to grayscale values (optional: multiply by 55 for enhanced contrast)
    # To save the original class indices, use seg directly
    mask_img = Image.fromarray(np.uint8(seg * 55))

    # Save the segmentation mask with the same filename as the original image
    mask_img.save(save_mask_root / img_path.name)