import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# --- Parameters ---
SIZE = 1000
JPEG_QUALITY = 20  # simulate heavy compression

# --- Step 1: Load image and resize ---
A = np.array(Image.open("input.png").convert("L").resize((SIZE, SIZE)), dtype=np.float32)

# --- Step 2: Generate orthogonal key ---
Q, _ = np.linalg.qr(np.random.randn(SIZE, SIZE).astype(np.float32))
K = Q  # orthogonal

# --- Step 3: Compute lock ---
X = np.linalg.inv(K) @ A

# --- Step 4: Scale to 0-255 ---
def scale_to_uint8(matrix, dithering=False):
    min_val, max_val = matrix.min(), matrix.max()
    scaled = (matrix - min_val) / (max_val - min_val) * 255
    if dithering:
        scaled += np.random.uniform(-0.5, 0.5, size=scaled.shape)
    return np.clip(scaled, 0, 255).astype(np.uint8), min_val, max_val

# --- Optional: simulate compression ---
def compress_image_jpeg(matrix_uint8, quality=JPEG_QUALITY):
    img = Image.fromarray(matrix_uint8, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    return np.array(img_compressed, dtype=np.uint8)

# --- Without dithering ---
X_no_dither, min_X, max_X = scale_to_uint8(X, dithering=False)
X_no_dither_compressed = compress_image_jpeg(X_no_dither)
X_no_dither_float = X_no_dither_compressed / 255 * (max_X - min_X) + min_X
A_rec_no_dither = K @ X_no_dither_float

# --- With dithering ---
X_dither, min_Xd, max_Xd = scale_to_uint8(X, dithering=True)
X_dither_compressed = compress_image_jpeg(X_dither)
X_dither_float = X_dither_compressed / 255 * (max_Xd - min_Xd) + min_Xd
A_rec_dither = K @ X_dither_float

# --- Visualization ---
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0,0].imshow(A, cmap='gray')
axs[0,0].set_title("Original A")
axs[0,1].imshow(X_no_dither_compressed, cmap='gray')
axs[0,1].set_title("Lock no dithering (compressed)")
axs[0,2].imshow(A_rec_no_dither, cmap='gray')
axs[0,2].set_title("Reconstructed no dithering")

axs[1,0].imshow(A, cmap='gray')
axs[1,0].set_title("Original A")
axs[1,1].imshow(X_dither_compressed, cmap='gray')
axs[1,1].set_title("Lock with dithering (compressed)")
axs[1,2].imshow(A_rec_dither, cmap='gray')
axs[1,2].set_title("Reconstructed with dithering")

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
