# decomposer_signed_auto_key.py
import numpy as np
from PIL import Image
import os

SIZE = 1024
EPS = 1e-12
KEY_FILE = "key_uni.png"
KEY_META = "key_min_max.npy"
LOCK_FILE = "lock.png"

def random_orthogonal_matrix(n=SIZE):
    A = np.random.randn(n, n).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q

def save_matrix_png_signed(matrix, filename):
    mat = np.array(matrix, dtype=np.float32)
    mn = float(mat.min())
    mx = float(mat.max())
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    norm = (mat - mn) / (rng + EPS)
    signed = norm * 255.0 - 128.0
    stored = np.clip(signed + 128.0, 0, 255).astype(np.uint8)
    if stored.ndim == 2:
        Image.fromarray(stored, mode="L").save(filename)
    else:
        Image.fromarray(stored).save(filename)
    return mn, mx

def load_matrix_png_signed(filename, min_val, max_val):
    arr = np.array(Image.open(filename))
    arr = arr.astype(np.float32)
    signed = arr - 128.0
    norm = (signed + 128.0) / 255.0
    mat = norm * (max_val - min_val) + min_val
    return mat

if __name__ == "__main__":
    A = np.array(Image.open("input.png").convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)

    if os.path.exists(KEY_FILE) and os.path.exists(KEY_META):
        K_min, K_max = np.load(KEY_META)
        K = load_matrix_png_signed(KEY_FILE, K_min, K_max)
    else:
        K = random_orthogonal_matrix(SIZE)
        K_min, K_max = save_matrix_png_signed(K, KEY_FILE)
        np.save(KEY_META, np.array([K_min, K_max], dtype=np.float32))

    X = np.empty_like(A, dtype=np.float32)
    K_inv = np.linalg.inv(K).astype(np.float32)
    for c in range(3):
        X[:, :, c] = K_inv @ A[:, :, c]

    l_min, l_max = save_matrix_png_signed(X, LOCK_FILE)

    K_rec = load_matrix_png_signed(KEY_FILE, K_min, K_max)
    X_rec = load_matrix_png_signed(LOCK_FILE, l_min, l_max)
    A_rec = np.empty_like(A, dtype=np.float32)
    for c in range(3):
        A_rec[:, :, c] = K_rec @ X_rec[:, :, c]
    A_rec = np.clip(A_rec, 0, 255).astype(np.uint8)
    Image.fromarray(A_rec).save("reconstructed.png")
