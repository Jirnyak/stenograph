import numpy as np
from PIL import Image

SIZE = 1000

def random_orthogonal_matrix(n=SIZE):
    """Generate well-conditioned invertible matrix using QR decomposition"""
    A = np.random.randn(n, n).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q

def save_matrix_png_float(matrix, filename):
    """Scale float matrix to 0-255 and save as PNG (RGB)"""
    # matrix shape: (SIZE, SIZE, 3)
    scaled = np.empty_like(matrix, dtype=np.uint8)
    min_vals = []
    max_vals = []
    for c in range(3):
        min_val, max_val = matrix[:,:,c].min(), matrix[:,:,c].max()
        scaled[:,:,c] = np.clip((matrix[:,:,c] - min_val)/(max_val - min_val) * 255, 0, 255).astype(np.uint8)
        min_vals.append(min_val)
        max_vals.append(max_val)
    Image.fromarray(scaled, mode="RGB").save(filename)
    return min_vals, max_vals

def load_image_float_color(filename, size=(SIZE,SIZE)):
    img = Image.open(filename).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)
    return arr  # shape: (SIZE, SIZE, 3)

# --- Step 1: Load image ---
A = load_image_float_color("input.png")

# --- Step 2: Generate stable invertible key ---
K = random_orthogonal_matrix(SIZE)

# --- Step 3: Compute lock matrix X per channel ---
X = np.empty_like(A)
for c in range(3):
    X[:,:,c] = np.linalg.inv(K) @ A[:,:,c]

# --- Step 5: Save key and lock as PNG ---
key_min, key_max = save_matrix_png_float(np.stack([K]*3, axis=-1), "key.png")  # duplicate key for RGB
lock_min, lock_max = save_matrix_png_float(X, "lock.png")

# --- Save scaling info ---
np.save("key_scale.npy", np.array([key_min, key_max], dtype=np.float32))
np.save("lock_scale.npy", np.array([lock_min, lock_max], dtype=np.float32))

# --- Optional: reconstruct for verification ---
A_rec = np.empty_like(A)
for c in range(3):
    A_rec[:,:,c] = K @ X[:,:,c]
A_rec = np.clip(A_rec, 0, 255)
Image.fromarray(A_rec.astype(np.uint8), mode="RGB").save("reconstructed.png")

print("Compression-resistant cipher done. Reconstructed image saved as reconstructed.png (color)")
