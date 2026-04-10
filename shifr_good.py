import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import resample

filename = "input.mp3"
matrix_size = 1024      
output_amp = "amp.png"
output_phase = "phase.png"

y, sr = librosa.load(filename, sr=None, mono=True)
samples_total = matrix_size * matrix_size
if len(y) < samples_total:
    y = np.pad(y, (0, samples_total - len(y)))
else:
    y = y[:samples_total]

frames = y.reshape(matrix_size, matrix_size)

fft_matrix = np.fft.fft(frames, axis=1)
mag = np.abs(fft_matrix)
phase = np.angle(fft_matrix)

mag_img = ((mag / np.max(mag)) * 255).astype(np.uint8)
phase_img = (((phase + np.pi) / (2*np.pi)) * 255).astype(np.uint8)

plt.imsave(output_amp, mag_img.T, cmap="gray")  
plt.imsave(output_phase, phase_img.T, cmap="gray")

mag_log = np.log1p(mag)
mag_log_img = ((mag_log / np.max(mag_log)) * 255).astype(np.uint8)
plt.imsave("amp_log.png", mag_log_img.T, cmap="gray")

