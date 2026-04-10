import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import resample

# -----------------------------
# PARAMETERS
# -----------------------------
filename = "input.mp3"
num_frames = 1000
target_samples = 1_000_000
top_N = 500               # keep more frequencies for better sound
output_amp = "amp.png"
output_phase = "phase.png"

# -----------------------------
# LOAD AUDIO
# -----------------------------
y, sr = librosa.load(filename, sr=None, mono=True)
samples_60s = sr*60
y = y[:samples_60s]

# -----------------------------
# RESAMPLE TO EXACTLY 1,000,000 SAMPLES
# -----------------------------
y = resample(y, target_samples)

# -----------------------------
# SPLIT INTO FRAMES
# -----------------------------
frame_size = target_samples // num_frames
frames = y[:frame_size*num_frames].reshape(num_frames, frame_size)

# -----------------------------
# FFT PER FRAME
# -----------------------------
fft_matrix = np.fft.fft(frames, axis=1)
mag = np.abs(fft_matrix)
phase = np.angle(fft_matrix)

# -----------------------------
# KEEP TOP N FREQUENCIES
# -----------------------------
for i in range(num_frames):
    idx = np.argsort(mag[i])[::-1][top_N:]
    fft_matrix[i, idx] = 0
    mag[i, idx] = np.abs(fft_matrix[i, idx])
    phase[i, idx] = np.angle(fft_matrix[i, idx])

# -----------------------------
# SAVE AMPLITUDE IMAGE
# -----------------------------
mag_matrix = np.abs(fft_matrix).T
mag_img = ((mag_matrix / np.max(mag_matrix))*255).astype(np.uint8)
plt.imsave(output_amp, mag_img, cmap="gray")

# -----------------------------
# SAVE PHASE IMAGE
# -----------------------------
phase_matrix = np.angle(fft_matrix).T
phase_img = (((phase_matrix + np.pi)/ (2*np.pi))*255).astype(np.uint8)
plt.imsave(output_phase, phase_img, cmap="gray")

print("Saved amplitude and phase images for 60s audio.")
