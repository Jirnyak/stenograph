#!/usr/bin/env python3
"""
2D визуализация падения объекта на ЧД (внешний наблюдатель).

- Объект НЕ уменьшается
- Красный сдвиг фотонов
- Кольцевой физический blur
- Физический zoom для наглядности (число)
"""
import os, math
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
import numpy as np

# ---------- CONFIG ----------
OBJ_PATH = "object.png"
BG_PATH  = "background.png"
OUT_DIR  = "output_frames"
OUT_GIF  = "output.gif"

M_solar = 1e6
FPS = 30
DURATION = 300
N_FRAMES = FPS * DURATION
BLUR_MAX = 12.0
r0_factor = 100.0  # начальное расстояние от горизонта
# -----------------------------

G = 6.67430e-11
c = 299792458
M_SUN = 1.98847e30

os.makedirs(OUT_DIR, exist_ok=True)

obj = Image.open(OBJ_PATH).convert("RGBA")
bg  = Image.open(BG_PATH).convert("RGBA")
W, H = obj.size
bg = bg.resize((W,H), Image.LANCZOS)

R_s = 2 * G * M_solar * M_SUN / c**2
r0 = r0_factor * R_s

try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# ------------------ Функции ------------------
def r_from_t_obs(t_obs):
    tau_obs = 2 * G * M_solar * M_SUN / c**3 * math.log(r0/R_s)
    return R_s + (r0 - R_s) * math.exp(-t_obs / tau_obs)

def intensity_and_redshift(r):
    g = math.sqrt(1 - R_s/r)
    intensity = g**4
    redshift = 1/g - 1.0
    return max(0.0, min(1.0, intensity)), redshift

def apply_redshift(img, z):
    r,g,b,a = img.split()
    if z < 3.0:
        factor = 1.0 + z
        r = r.point(lambda i: min(255,int(i*factor)))
        g = g.point(lambda i: int(i/(factor+1e-3)))
        b = b.point(lambda i: int(i/(factor+1e-3)))
    else:
        r = g = b = r.point(lambda _:0)
    return Image.merge("RGBA", (r,g,b,a))

def velocity_obs(r):
    return c * (1 - R_s/r) * math.sqrt(R_s/r)

def radial_blur_physical(img, r, R_s, BLUR_MAX):
    arr = np.array(img)
    h, w = arr.shape[:2]
    cx, cy = w//2, h//2

    blur_radius = BLUR_MAX * (R_s / max(r - R_s, 1e-6))**0.5
    blur_radius = int(max(1, blur_radius))

    y, x = np.indices((h, w))
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx**2 + dy**2)
    norm_dx = dx / (dist + 1e-6)
    norm_dy = dy / (dist + 1e-6)

    new_arr = arr.copy()
    for dr in range(1, blur_radius+1):
        shift_x = (norm_dx * dr).astype(int)
        shift_y = (norm_dy * dr).astype(int)
        x_new = np.clip(x + shift_x, 0, w-1)
        y_new = np.clip(y + shift_y, 0, h-1)
        new_arr = ((new_arr.astype(np.uint16) + arr[y_new, x_new].astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(new_arr)

# ------------------ Рендер кадра ------------------
def render_frame(obj_img, bg_img, r, t_obs):
    dR = r - R_s
    intensity, z = intensity_and_redshift(r)

    # Физический zoom как число
    zoom = (r0 - R_s) / max(dR, 1e-6)  # растёт при падении

    # объект без изменения размеров
    im_obj = obj_img.copy()
    enhancer = ImageEnhance.Brightness(im_obj)
    im_obj = enhancer.enhance(intensity)
    im_obj = apply_redshift(im_obj, z)

    # фон с zoom
    bg_zoomed_w, bg_zoomed_h = int(bg_img.width * zoom), int(bg_img.height * zoom)
    bg_zoomed = bg_img.resize((bg_zoomed_w, bg_zoomed_h), Image.LANCZOS)
    # вырезаем центр, чтобы вернуться к исходному размеру окна
    left = (bg_zoomed_w - bg_img.width)//2
    top  = (bg_zoomed_h - bg_img.height)//2
    bg_crop = bg_zoomed.crop((left, top, left + bg_img.width, top + bg_img.height))

    # холст для объекта
    obj_canvas = Image.new("RGBA", bg_img.size, (0,0,0,0))
    cx, cy = (bg_img.width - obj_img.width)//2, (bg_img.height - obj_img.height)//2
    obj_canvas.paste(im_obj, (cx, cy), im_obj.split()[3])

    # кольцевой blur
    obj_canvas = radial_blur_physical(obj_canvas, r, R_s, BLUR_MAX)

    # наложение на фон
    canvas = Image.alpha_composite(bg_crop.convert("RGBA"), obj_canvas)

    draw = ImageDraw.Draw(canvas)
    draw.text((5,5), f"M = 10^{int(round(math.log10(M_solar)))} suns", fill=(255,165,0,255), font=font)
    draw.text((5,90), f"t obs = {t_obs:.2f} s", fill=(255,165,0,255), font=font)
    draw.text((5,100), f"dR = {dR/1e3:.2f} km", fill=(255,165,0,255), font=font)
    v = velocity_obs(r)
    draw.text((5,110), f"v = {v/1e3:.2f} km/s" if v>=1e3 else f"v = {v:.2f} m/s", fill=(255,165,0,255), font=font)
    draw.text((5,120), f"zoom x{zoom:.2f}", fill=(0,255,0,255), font=font)

    return canvas

# ------------------ Генерация GIF ------------------
dt = DURATION / N_FRAMES
filenames = []
for i in tqdm(range(N_FRAMES), desc="Rendering frames"):
    t_obs = i * dt
    r = r_from_t_obs(t_obs)
    frame = render_frame(obj, bg, r, t_obs)
    fname = os.path.join(OUT_DIR, f"frame_{i:04d}.png")
    frame.save(fname)
    filenames.append(fname)

# assemble GIF
print("Assembling GIF...")
images = [imageio.imread(f) for f in filenames]
imageio.mimsave(OUT_GIF, images, fps=FPS)
print("Done.")
print(f"Frames in '{OUT_DIR}/', GIF -> '{OUT_GIF}'")
