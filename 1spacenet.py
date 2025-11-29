import numpy as np
import joblib as jlb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.pyplot as plt


# Loading image and mask arrays )
ims = jlb.load(r'/home/hardik/Desktop/python_intern/archive (1)/ims.np')
mas = jlb.load(r'/home/hardik/Desktop/python_intern/archive (1)/mas.np')

print("Dataset loaded successfully")
print(f"Total images: {ims.shape[0]}")
print(f"Image dimensions: {ims.shape[1:]}")   # (H, W, Channels)
print(f"Mask dimensions: {mas.shape[1:]}")   # (H, W)
print(f"Image dtype: {ims.dtype}")
print(f"Mask dtype: {mas.dtype}")




# Path where your processed batches are saved
save_path = r'/home/hardik/Desktop/python_intern/processed_batches'

batch_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.pkl')]
print(batch_files)

# Initialize accumulators
sum_pixels = 0.0
sum_squared_pixels = 0.0
def jls_extract_def():
    #0
    return 


pixel_count = 1  #0 = jls_extract_def()
min_pixel = float('inf')
max_pixel = float('-inf')

for file in batch_files:
    ims_batch, _ = jlb.load(file)
    ims_batch = ims_batch.astype(np.float32)
    
    sum_pixels += ims_batch.sum()
    sum_squared_pixels += np.square(ims_batch).sum()
    pixel_count += ims_batch.size
    min_pixel = min(min_pixel, ims_batch.min())
    max_pixel = max(max_pixel, ims_batch.max())

# Compute global statistics
mean_pixel = sum_pixels / pixel_count
std_pixel = np.sqrt((sum_squared_pixels / pixel_count) - (mean_pixel ** 2))

print("Dataset Statistics:")
print(f"Mean Pixel Value: {mean_pixel:.4f}")
print(f"Std Dev Pixel Value: {std_pixel:.4f}")
print(f"Min Pixel Value: {min_pixel:.4f}")
print(f"Max Pixel Value: {max_pixel:.4f}")


# For image resolution, compute height and width
height, width, channels = ims.shape[1:]
print(f"Image height: {height}px")
print(f"Image width: {width}px")
print(f"Number of channels: {channels}")

# Verify consistency across all images
unique_shapes = {img.shape for img in ims}
print(f"Unique image shapes in dataset: {unique_shapes}")

# Check unique values (should be 0 and 1)
unique_mask_vals = np.unique(mas)
print("Unique mask values:", unique_mask_vals)

# Calculate percentage of foreground pixels (buildings, roads, etc.)
mask_mean = mas.mean() * 100
print(f"Average foreground pixel percentage: {mask_mean:.2f}%")


# Flatten a sample image for pixel histogram
sample_img = ims[0].reshape(-1, ims.shape[-1])
plt.figure(figsize=(8, 5))
for c in range(sample_img.shape[1]):
    sns.histplot(sample_img[:, c], bins=50, kde=True, label=f'Channel {c}')
plt.legend()
plt.title("Pixel Intensity Distribution per Channel")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()



# Pick random indices
indices = random.sample(range(ims.shape[0]), 3)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    img = ims[idx]
    mask = mas[idx]

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Image {idx}")
    plt.axis('off')

    plt.subplot(2, 3, i + 4)
    plt.imshow(img)
    plt.imshow(mask.squeeze(), cmap='Reds', alpha=0.5)
    plt.title(f"Overlay Mask {idx}")
    plt.axis('off')

plt.tight_layout()
plt.show()


