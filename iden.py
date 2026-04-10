import numpy as np
from PIL import Image

SIZE = 1000        # image size
DIAG_WIDTH = 5     # thickness of diagonal in pixels
VAL = 255          # maximum value on diagonal (white)

def generate_identity_image(filename="identity.png"):
    """
    Generate a practical 'identity' image:
    - Fat white diagonal survives compression
    - Rest is gray (middle value)
    """
    img = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)  # gray background
    for i in range(SIZE):
        start = max(0, i - DIAG_WIDTH // 2)
        end   = min(SIZE, i + DIAG_WIDTH // 2 + 1)
        img[i, start:end, :] = VAL  # set diagonal stripe
    Image.fromarray(img, mode="RGB").save(filename)
    print(f"Identity image saved as {filename}")

if __name__ == "__main__":
    generate_identity_image()
