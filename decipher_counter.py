import numpy as np
from PIL import Image
import os

SIZE = 1024
STATE_PNG = "state.png"
EPS = 1e-12

def load_image_float_signed(filename):
    img = np.array(Image.open(filename).convert("RGB").resize((SIZE, SIZE)), dtype=np.uint8)
    signed = img.astype(np.float32) - 128.0 
    return signed

def _is_internal_identity(mat, tol=1e-6):
    if mat.ndim != 3:
        return False
    H, W, C = mat.shape
    if H != SIZE or W != SIZE:
        return False
    I2 = np.eye(SIZE, dtype=np.float32)
    for c in range(C):
        if not np.allclose(mat[:, :, c], I2, atol=tol):
            return False
    return True

def save_image_float_signed(matrix, filename):
    mat = np.array(matrix, dtype=np.float32)

    if _is_internal_identity(mat):
        stored = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)  
        idx = np.arange(SIZE)
        stored[idx, idx, :] = 255  
        Image.fromarray(stored).save(filename)
        return

    mn = mat.min()
    mx = mat.max()
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    norm = (mat - mn) / (rng + EPS)
    stored = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(stored).save(filename)


def generate_identity():
    I = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for c in range(3):
        np.fill_diagonal(I[:, :, c], 1.0)
    return I

def is_flat_image(img, tol=1e-6):
    return all(img[:, :, c].ptp() < tol for c in range(3))

def update_chain(new_file):
    X_new = load_image_float_signed(new_file)

    if not os.path.exists(STATE_PNG):
        save_image_float_signed(X_new, STATE_PNG)
        return

    X_old = load_image_float_signed(STATE_PNG)

    A_rec = np.empty_like(X_new, dtype=np.float32)
    for c in range(3):
        A_rec[:, :, c] = X_new[:, :, c] @ X_old[:, :, c]

    if is_flat_image(A_rec):
        A_rec = generate_identity()

    save_image_float_signed(A_rec, STATE_PNG)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        update_chain(sys.argv[1])
    else:
        update_chain("new.png")
