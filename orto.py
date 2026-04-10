import numpy as np
from PIL import Image

SIZE = 1000  # final square size

def load_image_rgb_signed(filename, size=(SIZE, SIZE)):
    """Load image, resize, convert to float with -128..127 range"""
    img = Image.open(filename).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)
    return arr - 128  # shift to signed

def save_image_rgb_signed(matrix, filename):
    """Shift back to 0-255 and save as PNG"""
    arr = np.clip(matrix + 128, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(filename)

def orthogonalize_image_color(img, epsilon=1e-3):
    """
    Make each color channel invertible using QR decomposition
    Preserves visual similarity by adding tiny noise before orthogonalization.
    """
    H, W, C = img.shape
    out = np.empty_like(img)
    for c in range(C):
        channel = img[:, :, c]
        channel += np.random.uniform(-epsilon, epsilon, size=(H, W))
        Q, R = np.linalg.qr(channel)
        scale = np.linalg.norm(channel) / np.linalg.norm(Q)
        out[:, :, c] = Q * scale
    return out

def pseudo_inverse_color(img):
    """Compute pseudo-inverse for each color channel"""
    H, W, C = img.shape
    inv = np.empty((W, H, C), dtype=np.float32)  # note shape swap for pseudo-inverse
    for c in range(C):
        inv[:, :, c] = np.linalg.pinv(img[:, :, c])
    return inv

if __name__ == "__main__":
    # --- Load original image ---
    img = load_image_rgb_signed("input.png")

    # --- Orthogonalize (make invertible) ---
    img_ortho = orthogonalize_image_color(img, epsilon=1e-3)
    save_image_rgb_signed(img_ortho, "input_invertible.png")
    print("Orthogonalized invertible image saved as input_invertible.png")

    # --- Compute pseudo-inverse of orthogonal image ---
    img_inv = pseudo_inverse_color(img_ortho)
    save_image_rgb_signed(img_inv, "input_inverse.png")
    print("Pseudo-inverse image saved as input_inverse.png")

    # --- Test reconstruction (orthogonal × pseudo-inverse → identity-ish) ---
    rec = np.empty_like(img)
    for c in range(3):
        rec[:, :, c] = img_ortho[:, :, c] @ img_inv[:, :, c]
    save_image_rgb_signed(rec, "reconstruction.png")
    print("Reconstruction (should be close to identity) saved as reconstruction.png")

    # --- Optional: reconstruct original using orthogonal and pseudo-inverse ---
    final = np.empty_like(img)
    for c in range(3):
        final[:, :, c] = img[:, :, c] @ img_inv[:, :, c]  # original × pseudo-inverse of ortho
    save_image_rgb_signed(final, "final_multiplied.png")
    print("Original multiplied by pseudo-inverse of orthogonal saved as final_multiplied.png")
