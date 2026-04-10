# lemonlock_kron.py
import numpy as np
from PIL import Image
import os

SIZE = 512            # base working size
FACTOR = 2            # expansion factor (each pixel -> FACTOR×FACTOR block)
MARGIN = 16           # keep values inside [16..240] to resist compression
EPS = 1e-12
KEY_FILE = "key_test.png"
KEY_META = "key_test.npy"
LOCK_FILE = "lock.png"

def random_orthogonal_matrix(n=SIZE):
    A = np.random.randn(n, n).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q

def kron_expand(mat, factor=FACTOR, mode="image"):
    """Kronecker expansion for key or image."""
    if mode == "key":
        return np.kron(mat, np.eye(factor, dtype=np.float32))
    elif mode == "image":
        return np.kron(mat, np.ones((factor, factor), dtype=np.float32))
    else:
        raise ValueError("mode must be 'key' or 'image'")

def save_matrix_png_signed(matrix, filename, margin=5):
    """Save matrix as PNG with safe margins."""
    mat = np.array(matrix, dtype=np.float32)
    mn, mx = float(mat.min()), float(mat.max())
    rng = mx - mn if mx > mn else 1.0
    norm = (mat - mn) / rng
    stored = (norm * (255 - 2*margin) + margin).astype(np.uint8)

    if stored.ndim == 2:
        img = Image.fromarray(stored, mode="L")
    else:
        img = Image.fromarray(stored, mode="RGB")

    img.save(filename, quality=95)
    return mn, mx

def load_matrix_png_signed(filename, min_val, max_val, is_key=False):
    """Load matrix from PNG and map back to float values."""
    arr = np.array(Image.open(filename)).astype(np.float32)
    norm = (arr - arr.min()) / (arr.max() - arr.min() + EPS)
    mat = norm * (max_val - min_val) + min_val
    if is_key and mat.ndim == 3:  # take single channel if key stored as RGB
        mat = mat[:, :, 0]
    return mat

if __name__ == "__main__":
    # 1. Load input image (small size)
    A = np.array(Image.open("input_test.png").convert("RGB").resize((SIZE, SIZE)),
                 dtype=np.float32)

    # 2. Load or generate key
    if os.path.exists(KEY_FILE) and os.path.exists(KEY_META):
        K_min, K_max = np.load(KEY_META)
        K_small = load_matrix_png_signed(KEY_FILE, K_min, K_max, is_key=True)
    else:
        K_small = random_orthogonal_matrix(SIZE)
        K_min, K_max = save_matrix_png_signed(K_small, KEY_FILE)
        np.save(KEY_META, np.array([K_min, K_max], dtype=np.float32))

    # 3. Expand key and image
    K_big = kron_expand(K_small, FACTOR, mode="key")
    A_big = np.empty((SIZE*FACTOR, SIZE*FACTOR, 3), dtype=np.float32)
    for c in range(3):
        A_big[:, :, c] = kron_expand(A[:, :, c], FACTOR, mode="image")

    # 4. Encode lock
    K_big_inv = np.linalg.inv(K_big).astype(np.float32)
    X_big = np.empty_like(A_big, dtype=np.float32)
    for c in range(3):
        X_big[:, :, c] = K_big_inv @ A_big[:, :, c]

    l_min, l_max = save_matrix_png_signed(X_big, LOCK_FILE)
    k_min, k_max = save_matrix_png_signed(K_big, KEY_FILE)

    # 5. Decode back for testing
    K_rec_small = load_matrix_png_signed(KEY_FILE, K_min, K_max, is_key=True)
    K_rec = kron_expand(K_rec_small, FACTOR, mode="key")
    X_rec = load_matrix_png_signed(LOCK_FILE, l_min, l_max)

    A_rec = np.empty_like(A_big, dtype=np.float32)
    for c in range(3):
        A_rec[:, :, c] = K_rec @ X_rec[:, :, c]

    A_rec = np.clip(A_rec, 0, 255).astype(np.uint8)

    # 6. Save reconstruction
    Image.fromarray(A_rec).save("reconstructed.png")
