import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# --- Load floating-point lock image ---
lock = np.array(Image.open("lock.png").convert("L"), dtype=np.float32)

# --- Normalize back to 0..1 ---
lock_norm = lock / 255.0

# --- Attempt 1: smooth out correlations ---
# Floating-point lock is linear transform of original image
# We try to decorrelate it using Gaussian blur + scaling
approx = gaussian_filter(lock_norm, sigma=2)

# --- Rescale to 0..255 ---
approx_img = np.clip(approx * 255, 0, 255).astype(np.uint8)

# --- Save result ---
Image.fromarray(approx_img, mode="L").save("approx_from_lock.png")
print("Saved approx_from_lock.png")
