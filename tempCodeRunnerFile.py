import numpy as np
import random
import os
from PIL import Image

# Load the numpy file
np_file_path = '/home/hardik/Desktop/python_intern/archive (1)/ims.np'
output_dir = '/home/hardik/Desktop/python_intern/spacenet_testdata'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the numpy array
print("Loading numpy file...")
images = np.load(np_file_path, allow_pickle=True)
print(f"Loaded array with shape: {images.shape}")

# Pick a random index
random_idx = random.randint(0, len(images) - 1)
print(f"Selected random image at index: {random_idx}")

# Get the random image
random_image = images[random_idx]

# Save the image
output_path = os.path.join(output_dir, f'random_image_{random_idx}.png')

# Handle different possible array formats
if random_image.dtype == np.uint8:
    img = Image.fromarray(random_image)
elif random_image.max() <= 1.0:
    # If normalized to [0, 1], scale to [0, 255]
    img = Image.fromarray((random_image * 255).astype(np.uint8))
else:
    img = Image.fromarray(random_image.astype(np.uint8))

img.save(output_path)
print(f"Image saved to: {output_path}")
print(f"Image shape: {random_image.shape}")