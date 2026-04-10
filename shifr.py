import numpy as np
from PIL import Image

SIZE = 1000
KEY_PNG = "key.png"
LOCK_PNG = "lock.png"

def load_image_float(filename):
    img = Image.open(filename).convert("RGB").resize((SIZE, SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def save_image_float(matrix, filename):
    """Save 0-1 float matrix as 0-255 PNG"""
    img = np.clip(matrix*255, 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(filename)

def random_orthogonal_matrix(n=SIZE):
    """Generate dense orthogonal matrix with values ±1"""
    A = np.random.randn(n, n).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    Q /= np.max(np.abs(Q))  # scale to ±1
    return Q

def decompose_image(image_file):
    A = load_image_float(image_file)
    K = random_orthogonal_matrix(SIZE)

    # compute lock per channel
    X = np.empty_like(A)
    for c in range(3):
        X[:,:,c] = np.linalg.inv(K) @ A[:,:,c]

    # normalize key and lock to 0–1 for PNG
    K_norm = (K + 1.0)/2  # scale ±1 → 0–1
    X_min = X.min()
    X_max = X.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)

    save_image_float(np.stack([K_norm]*3, axis=-1), KEY_PNG)
    save_image_float(X_norm, LOCK_PNG)
    print(f"✅ Decomposition done: key -> {KEY_PNG}, lock -> {LOCK_PNG}")

    return X_min, X_max  # return for reconstruction

def reconstruct_image(X_min, X_max):
    K_norm = load_image_float(KEY_PNG)[:,:,0]  # take one channel
    X_norm = load_image_float(LOCK_PNG)

    # recover original range
    K = K_norm*2 - 1
    X = X_norm*(X_max - X_min) + X_min

    A_rec = np.empty_like(X)
    for c in range(3):
        A_rec[:,:,c] = K @ X[:,:,c]

    # per-channel normalization
    for c in range(3):
        A_rec[:,:,c] /= (A_rec[:,:,c].max() + 1e-12)

    save_image_float(A_rec, "reconstructed.png")
    print("✅ Image reconstructed: reconstructed.png")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    X_min, X_max = decompose_image("input.png")
    reconstruct_image(X_min, X_max)
