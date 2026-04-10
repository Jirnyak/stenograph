import numpy as np
from PIL import Image
from scipy.linalg import hadamard

SIZE = 1024  # must be power of 2 for Hadamard

# ---------------- Hadamard Key ---------------- #
def hadamard_key(n=SIZE, seed=None):
    H = hadamard(n).astype(np.float32)
    H = H / np.sqrt(n)  # normalize

    # Random permutation to avoid flat first row/column
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    H = H[perm, :][:, perm]
    return H

# ---------------- Save / Load ---------------- #
def save_matrix_png(matrix, filename, bit16=False):
    """Save 2D or 3D matrix to PNG with scaling"""
    min_val = matrix.min()
    max_val = matrix.max()

    if bit16 and matrix.ndim == 2:
        # 16-bit grayscale
        scaled = np.clip((matrix - min_val) / (max_val - min_val + 1e-12) * 65535, 0, 65535).astype(np.uint16)
        mode = "I;16"
    else:
        # 8-bit RGB/Gray
        scaled = np.clip((matrix - min_val) / (max_val - min_val + 1e-12) * 255, 0, 255).astype(np.uint8)
        mode = "L" if matrix.ndim == 2 else "RGB"

    Image.fromarray(scaled, mode=mode if matrix.ndim == 2 else None).save(filename)
    return min_val, max_val

def load_matrix_png(filename, min_val, max_val, bit16=False, channels=1):
    arr = np.array(Image.open(filename), dtype=np.uint16 if bit16 else np.uint8).astype(np.float32)
    arr = arr / (65535.0 if bit16 else 255.0) * (max_val - min_val) + min_val
    if channels == 3 and arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

# ---------------- Main ---------------- #
# Load original image
A = np.array(Image.open("input.png").convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)

# Generate Hadamard orthogonal key (permuted)
K = hadamard_key(SIZE, seed=42)
K_inv = K.T

# Compute lock
X = np.empty_like(A)
for c in range(3):
    X[:, :, c] = K_inv @ A[:, :, c]

# Save key (16-bit grayscale) and lock (8-bit RGB)
k_min, k_max = save_matrix_png(K, "key.png", bit16=True)
l_min, l_max = save_matrix_png(X, "lock.png", bit16=False)

# Load back
K_rec = load_matrix_png("key.png", k_min, k_max, bit16=True, channels=1)
X_rec = load_matrix_png("lock.png", l_min, l_max, bit16=False, channels=3)

# Reconstruct original image
A_rec = np.empty_like(A)
for c in range(3):
    A_rec[:, :, c] = K_rec @ X_rec[:, :, c]

A_rec = np.clip(A_rec, 0, 255).astype(np.uint8)
Image.fromarray(A_rec).save("reconstructed.png")

print("✅ Reconstruction successful")
