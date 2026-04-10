import numpy as np
import librosa
from PIL import Image
from scipy.signal import resample

filename = "input.mp3"
matrix_size = 1024
sr_target = 44100
top_N = 256
out_png = "shifr.png"

y, sr_orig = librosa.load(filename, sr=None, mono=True)
duration_sec = max(int(len(y) / sr_orig), 1)
samples_target = sr_target * duration_sec

if len(y) < samples_target:
    y = np.pad(y, (0, samples_target - len(y)))
else:
    y = y[:samples_target]

total_samples = matrix_size * matrix_size
y = resample(y, total_samples)
frames = y.reshape(matrix_size, matrix_size)

fft_matrix = np.fft.fft(frames, axis=1)
mag = np.abs(fft_matrix)
phase = np.angle(fft_matrix)

mag_log = np.log1p(mag)
max_log = np.max(mag_log) if np.max(mag_log)>0 else 1.0
mag_signed = np.clip(mag_log / max_log * 256 - 128, -128, 127).astype(np.int8)

phase_signed = np.clip((phase / np.pi) * 128, -128, 127).astype(np.int8)

b_channel = np.full((matrix_size, matrix_size), int(np.clip(duration_sec,0,255)), dtype=np.uint8)

rgb = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)
rgb[:, :, 0] = (mag_signed.T + 128).astype(np.uint8)
rgb[:, :, 1] = (phase_signed.T + 128).astype(np.uint8)
rgb[:, :, 2] = b_channel

Image.fromarray(rgb, mode="RGB").save(out_png)

mag_rgb = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)
mag_rgb[:, :, 0] = (mag_signed.T + 128).astype(np.uint8)
Image.fromarray(mag_rgb, mode="RGB").save("amplitude.png")

phase_rgb = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)
phase_rgb[:, :, 1] = (phase_signed.T + 128).astype(np.uint8)
Image.fromarray(phase_rgb, mode="RGB").save("phase.png")