import numpy as np
from PIL import Image
import os

SIZE = 500
MOD = 251
STATE_PNG = "lock.png"

# ---------------- Load / Save ---------------- #
def load_image_mod(filename):
    img = np.array(Image.open(filename).convert("L").resize((SIZE, SIZE)), dtype=np.int64)
    return img % MOD

def save_image_mod(matrix, filename):
    scaled = (matrix % MOD * 255 // (MOD-1)).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(filename)

# ---------------- Identity ---------------- #
def generate_identity():
    I = np.zeros((SIZE, SIZE), dtype=np.int64)
    np.fill_diagonal(I, 1)
    return I

# ---------------- Modular Multiplication ---------------- #
def mod_matmul(A, B, mod):
    """Matrix multiplication mod `mod` avoiding overflow"""
    n = A.shape[0]
    C = np.zeros_like(A, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += (A[i,k] * B[k,j]) % mod
            C[i,j] = s % mod
    return C

# ---------------- Modular Chain ---------------- #
def update_chain(new_file):
    X_new = load_image_mod(new_file)

    if not os.path.exists(STATE_PNG):
        save_image_mod(X_new, STATE_PNG)
        return

    X_old = load_image_mod(STATE_PNG)

    # Markovian modular multiplication
    A_rec = mod_matmul(X_new, X_old, MOD)

    # Prevent collapsing to zero
    if np.all(A_rec == 0):
        A_rec = generate_identity()

    save_image_mod(A_rec, STATE_PNG)

# ---------------- Main ---------------- #
if __name__ == "__main__":
    update_chain("key.png")
