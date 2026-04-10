import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

matrix_size = 1024
input_amp = "amp.png"
input_phase = "phase.png"
output_audio = "reconstructed.wav"
sr = 44100 

samples_total = matrix_size * matrix_size

amp_img = plt.imread(input_amp)
phase_img = plt.imread(input_phase)

if amp_img.ndim == 3: 
    amp_img = amp_img[...,0]
if phase_img.ndim == 3:
    phase_img = phase_img[...,0]

mag_matrix = amp_img.astype(np.float32).reshape(matrix_size, matrix_size) / 255.0
phase_matrix = (phase_img.astype(np.float32).reshape(matrix_size, matrix_size) / 255.0) * 2*np.pi - np.pi

fft_matrix = mag_matrix.T * np.exp(1j * phase_matrix.T)
frames_rec = np.fft.ifft(fft_matrix, axis=1).real
y_rec = frames_rec.reshape(-1)

y_rec /= np.max(np.abs(y_rec))
y_rec *= 0.9

sf.write(output_audio, y_rec, sr)

