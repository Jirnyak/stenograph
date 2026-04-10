# python3.11 decipher.py
import numpy as np
from PIL import Image
from sympy import Matrix

MOD = 251

# --- Load key and lock images ---
K = np.array(Image.open("key.png"), dtype=np.uint16)
X = np.array(Image.open("lock.png"), dtype=np.uint16)

# --- Reconstruct original image ---
A_rec = (Matrix(K.tolist()) * Matrix(X.tolist())) % MOD
A_rec = np.array(A_rec.tolist(), dtype=np.uint16)

# --- Save reconstructed image ---
Image.fromarray(A_rec.astype(np.uint8), mode="L").save("deciphered.png")
print("Deciphered image saved as deciphered.png")
