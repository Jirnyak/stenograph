import numpy as np
from PIL import Image
import soundfile as sf
from scipy.signal import resample

matrix_size = 1024
sr_target = 44100
in_png = "shifr.png"
out_wav = "reconstructed.wav"

img = Image.open(in_png)
rgb = np.array(img)

mag_signed = rgb[:,:,0].T.astype(np.float32)-128
phase_signed = rgb[:,:,1].T.astype(np.float32)-128
b_channel = rgb[:,:,2].T.astype(np.float32)

duration_sec = max(int(np.mean(b_channel)),1)
samples_target = sr_target * duration_sec

max_mag = np.max(mag_signed+128)
mag = np.expm1((mag_signed+128)/256*np.log1p(max_mag))
phase = (phase_signed/128)*np.pi

fft_matrix = mag*np.exp(1j*phase)
frames_rec = np.fft.ifft(fft_matrix, axis=1).real
y_rec = frames_rec.reshape(-1)

y_rec = resample(y_rec, samples_target)
sf.write(out_wav, y_rec, sr_target)
