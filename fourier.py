import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample

# -----------------------------
# PARAMETERS
# -----------------------------
filename = "input.mp3"
num_frames = 1000             # columns in matrix (time axis)
target_samples = 1_000_000    # total samples after resampling
top_N = 200                   # keep strongest frequencies per frame
output_image = "spectrogram_1000x1000.png"
output_audio = "reconstructed.wav"

# -----------------------------
# LOAD AUDIO
# -----------------------------
y, sr = librosa.load(filename, sr=None, mono=True)

# Cut first 1 minute
samples_1min = sr * 60
y = y[:samples_1min]

# -----------------------------
# RESAMPLE TO EXACTLY 1,000,000 SAMPLES
# -----------------------------
y = resample(y, target_samples)
total_samples = len(y)  # now 1_000_000

# -----------------------------
# SPLIT INTO FRAMES (1000 frames)
# -----------------------------
frame_size = total_samples // num_frames  # = 1000
frames = y.reshape(num_frames, frame_size)  # shape: 1000x1000

# -----------------------------
# FFT PER FRAME
# -----------------------------
fft_matrix = np.fft.fft(frames, axis=1)
mag = np.abs(fft_matrix)

# -----------------------------
# KEEP TOP N FREQUENCIES PER FRAME
# -----------------------------
for i in range(num_frames):
    idx = np.argsort(mag[i])[::-1][top_N:]
    fft_matrix[i, idx] = 0

# -----------------------------
# SAVE MATRIX AS PNG
# -----------------------------
mag_matrix = np.abs(fft_matrix).T  # shape: freq x frames = 1000x1000

# Normalize to [0,255]
mag_matrix = mag_matrix / np.max(mag_matrix) * 128.0
mag_matrix = np.clip(mag_matrix, -128, 128)
img = ((mag_matrix + 128) / 256 * 255).astype(np.uint8)
plt.imsave(output_image, img, cmap="gray")
print(f"Saved 1000x1000 spectrogram image: {output_image}")

# -----------------------------
# RECONSTRUCT AUDIO
# -----------------------------
# Keep Hermitian symmetry for IFFT
top = fft_matrix[:, 1:frame_size//2]
conj_part = np.conj(np.flip(top, axis=1))
fft_full = np.hstack([fft_matrix, conj_part])
frames_rec = np.fft.ifft(fft_full, axis=1).real
y_rec = frames_rec.reshape(-1)

# Resample reconstructed audio to exactly 1 minute
y_rec_resampled = resample(y_rec, samples_1min)

sf.write(output_audio, y_rec_resampled, sr)
print(f"Saved reconstructed audio: {output_audio} (exactly 1 minute)")
