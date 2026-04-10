import numpy as np
from PIL import Image

SIZE = 1000

def load_matrix_png_float_color(filename, min_vals, max_vals):
    """
    Load an RGB key/lock saved as PNG and rescale each channel separately
    min_vals, max_vals: list or array of 3 floats (for R,G,B)
    """
    arr = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)
    matrix = np.empty_like(arr)
    for c in range(3):
        matrix[:,:,c] = arr[:,:,c] / 255 * (max_vals[c] - min_vals[c]) + min_vals[c]
    return matrix  # shape: (SIZE, SIZE, 3)

# --- Load scaling info ---
key_min, key_max = np.load("key_scale.npy")  # each is array of 3
lock_min, lock_max = np.load("lock_scale.npy")

# --- Load key and lock images ---
K = load_matrix_png_float_color("key.png", key_min, key_max)
X = load_matrix_png_float_color("lock.png", lock_min, lock_max)

# --- Reconstruct image ---
A_rec = np.empty_like(X)
for c in range(3):
    A_rec[:,:,c] = K[:,:,c] @ X[:,:,c]  # channel-wise multiplication

A_rec = np.clip(A_rec, 0, 255)
Image.fromarray(A_rec.astype(np.uint8), mode="RGB").save("deciphered.png")

print("Deciphered image saved as deciphered.png")
