# lemonlock_expand.py
import numpy as np
from PIL import Image
import os

SIZE = 512      # base resolution
FACTOR = 2      # expansion factor
FINAL_SIZE = SIZE * FACTOR
EPS = 1e-12

KEY_FILE = "key_large.png"
KEY_META = "key_large.npy"
LOCK_FILE = "lock.png"

def random_orthogonal_matrix(n=SIZE):
    A = np.random.randn(n, n).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q

def pixel_expand(matrix, factor=2):
    mat = np.array(matrix)
    if mat.ndim == 2:  # grayscale
        return np.repeat(np.repeat(mat, factor, axis=0), factor, axis=1)
    elif mat.ndim == 3 and mat.shape[2] == 3:  # RGB
        return np.repeat(np.repeat(mat, factor, axis=0), factor, axis=1)
    else:
        raise ValueError(f"Unsupported ndim for pixel_expand: {mat.ndim}")

def save_matrix_png_signed(matrix, filename):
    mat = np.array(matrix, dtype=np.float32)
    mn = float(mat.min())
    mx = float(mat.max())
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    norm = (mat - mn) / (rng + EPS)
    stored = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    if stored.ndim == 2:
        Image.fromarray(stored, mode="L").save(filename)
    elif stored.ndim == 3 and stored.shape[2] == 3:
        Image.fromarray(stored, mode="RGB").save(filename)
    else:
        raise ValueError(f"Unsupported array shape {stored.shape}")

    return mn, mx

def load_matrix_png_signed(filename, min_val, max_val):
    arr = np.array(Image.open(filename))
    arr = arr.astype(np.float32)
    norm = arr / 255.0
    mat = norm * (max_val - min_val) + min_val
    return mat

if __name__ == "__main__":
    A = np.array(Image.open("input.png").convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)

    if os.path.exists(KEY_FILE) and os.path.exists(KEY_META):
        K_min, K_max = np.load(KEY_META)
        K = load_matrix_png_signed(KEY_FILE, K_min, K_max)
    else:
        K = random_orthogonal_matrix(SIZE)
        K_big = pixel_expand(K, FACTOR)
        K_min, K_max = save_matrix_png_signed(K, KEY_FILE)
        np.save(KEY_META, np.array([K_min, K_max], dtype=np.float32))

    X = np.empty_like(A, dtype=np.float32)
    K_inv = np.linalg.inv(K).astype(np.float32)
    for c in range(3):
        X[:, :, c] = K_inv @ A[:, :, c]

    A_big = pixel_expand(A, FACTOR)
    X_big = pixel_expand(X, FACTOR)

    l_min, l_max = save_matrix_png_signed(X_big, "lock.png")




