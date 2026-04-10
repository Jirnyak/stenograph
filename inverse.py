import numpy as np
from PIL import Image

SIZE = 1000  # size to resize

def load_image_float(filename, size=(SIZE,SIZE)):
    img = Image.open(filename).convert("RGB").resize(size)
    return np.array(img, dtype=np.float32)

def save_image_float(matrix, filename):
    matrix = np.clip(matrix, 0, 255).astype(np.uint8)
    Image.fromarray(matrix, mode="RGB").save(filename)

def compute_inverse_image(A):
    """Compute pseudo-inverse per color channel"""
    inv = np.empty_like(A)
    for c in range(3):
        channel = A[:,:,c]
        # Use pseudo-inverse to handle non-orthogonal / compressible matrices
        inv[:,:,c] = np.linalg.pinv(channel)
    return inv

# --- Load image ---
img = load_image_float("input_invertible_gray.png")

# --- Compute approximate inverse ---
img_inv = compute_inverse_image(img)

# --- Save result ---
save_image_float(img_inv, "inverse.png")
print("Inverse image saved as inverse.png")
