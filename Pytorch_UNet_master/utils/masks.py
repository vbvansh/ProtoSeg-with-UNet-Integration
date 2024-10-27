import os
from PIL import Image
import numpy as np
import pandas
mask_dir = 'Pytorch-UNet-master/data/masks'

for mask_file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_file)
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask_array = np.array(mask)

    # Ensure it's binary (0 and 1 only)
    mask_array[mask_array > 0] = 1
    
    # Save the binary mask back
    new_mask = Image.fromarray(mask_array.astype(np.uint8))
    new_mask.save(mask_path)
