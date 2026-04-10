import numpy as np
from PIL import Image

SIZE = 500
MOD = 251  # prime for modular arithmetic

# ---------------- Utility ---------------- #
def save_matrix_png(matrix, filename):
    """Save integer matrix mod MOD as grayscale PNG"""
    scaled = (matrix % MOD * 255 // (MOD-1)).astype(np.uint8)
    Image.fromarray(scaled, mode="L").save(filename)

def load_image_mod(filename):
    """Load grayscale PNG as integers mod MOD"""
    img = np.array(Image.open(filename).convert("L").resize((SIZE, SIZE)), dtype=np.int64)
    return img % MOD

# ---------------- Key Generation ---------------- #
def random_invertible_upper_triangular(n, mod):
    """Generate random upper-triangular invertible matrix mod `mod`"""
    while True:
        K = np.triu(np.random.randint(1, mod, size=(n,n), dtype=np.int64))
        if np.all(K.diagonal() != 0):
            return K

def mod_solve_upper_triangular(K, A, mod):
    """Solve K @ X = A mod `mod` for upper-triangular K"""
    n = K.shape[0]
    X = np.zeros_like(A, dtype=np.int64)
    for j in range(n):
        for i in reversed(range(n)):
            s = int(A[i,j])
            for k in range(i+1, n):
                s -= int(K[i,k]) * int(X[k,j])
            X[i,j] = (s * pow(int(K[i,i]), -1, mod)) % mod
    return X

# ---------------- Main ---------------- #
# Load grayscale input
A = load_image_mod("input.png")

# Generate key
K = random_invertible_upper_triangular(SIZE, MOD)

# Compute lock
X = mod_solve_upper_triangular(K, A, MOD)

# Save key and lock
save_matrix_png(K, "key.png")
save_matrix_png(X, "lock.png")

# Verify reconstruction
A_rec = (K @ X) % MOD
save_matrix_png(A_rec, "reconstructed.png")
print("✅ Reconstruction exact?", np.array_equal(A % MOD, A_rec))
