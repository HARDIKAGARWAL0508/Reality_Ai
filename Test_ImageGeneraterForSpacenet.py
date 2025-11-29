import numpy as np
import random
import os
import joblib as jlb
from PIL import Image

# File paths
np_file_path = '/home/hardik/Desktop/python_intern/archive (1)/ims.np'
output_dir = '/home/hardik/Desktop/python_intern/spacenet_testdata'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the file using joblib (same as your main script)
print("Loading images using joblib...")
images = jlb.load(np_file_path)
images = np.array(images)

print(f"Loaded array with shape: {images.shape}")
print(f"Array dtype: {images.dtype}")

# Pick a random index
random_idx = random.randint(0, len(images) - 1)
print(f"Selected random image at index: {random_idx}")

# Get the random image
random_image = images[random_idx]
print(f"Image shape: {random_image.shape}")

# Save the image
output_path = os.path.join(output_dir, f'random_image_{random_idx}.png')

# Handle different possible array formats
# Normalize to 0-255 range if needed
if random_image.dtype == np.float32 or random_image.dtype == np.float64:
    if random_image.max() <= 1.0:
        # If normalized to [0, 1], scale to [0, 255]
        random_image = (random_image * 255).astype(np.uint8)
    else:
        random_image = random_image.astype(np.uint8)
elif random_image.dtype != np.uint8:
    random_image = random_image.astype(np.uint8)

# Convert to PIL Image and save
# Handle different channel configurations
if random_image.ndim == 3:
    if random_image.shape[-1] == 1:
        # Single channel - convert to 2D for grayscale
        img = Image.fromarray(random_image[:, :, 0], mode='L')
    elif random_image.shape[-1] == 3:
        # RGB
        img = Image.fromarray(random_image, mode='RGB')
    elif random_image.shape[-1] == 4:
        # RGBA
        img = Image.fromarray(random_image, mode='RGBA')
    else:
        # Multiple channels - use first 3 as RGB
        img = Image.fromarray(random_image[:, :, :3], mode='RGB')
elif random_image.ndim == 2:
    # Already 2D grayscale
    img = Image.fromarray(random_image, mode='L')
else:
    raise ValueError(f"Unexpected image dimensions: {random_image.shape}")

img.save(output_path)
print(f"✅ Image saved successfully to: {output_path}")