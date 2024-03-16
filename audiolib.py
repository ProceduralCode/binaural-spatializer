
import scipy.io.wavfile as wavfile
from scipy.signal import iirfilter, sosfilt
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

sample_rate = 48000

def read(filename):
	global sample_rate
	sample_rate, audio = wavfile.read(filename)
	if audio.dtype.kind == 'f':
		return audio
	return audio / np.iinfo(audio.dtype).max    # normalize to [-1, 1]

def write(filename, audio):
	data = (audio * np.iinfo(np.int16).max).astype(np.int16)
	wavfile.write(filename, sample_rate, data)

def play(audio):
	data = (audio * np.iinfo(np.int16).max).astype(np.int16)
	sd.play(data, sample_rate)
	sd.wait()

def display(audio):
	plt.plot(audio)
	plt.show()

def pad_to_length(audio, length):
	if len(audio) >= length:
		return audio
	if len(audio.shape) == 1:
		return np.pad(audio, (0, length - len(audio)))
	return np.pad(audio, ((0, length - len(audio)), (0, 0)))

def combine(audio1, audio2):
	if len(audio1.shape) != len(audio2.shape):
		raise ValueError('Audio channels do not match')
	if audio1.shape[0] > audio2.shape[0]:
		audio2 = pad_to_length(audio2, audio1.shape[0])
	else:
		audio1 = pad_to_length(audio1, audio2.shape[0])
	return audio1 + audio2

def mix(audio1, audio2, ratio):
	return audio1 * ratio + audio2 * (1 - ratio)

def pass_filter(audio, cutoff, order=1, type='highpass'):
	sos = iirfilter(order, cutoff, btype=type, fs=sample_rate, output='sos')
	if len(audio.shape) == 2:
		return np.array([sosfilt(sos, audio[:, i]) for i in range(2)]).T
	return sosfilt(sos, audio)

def apply_impulse_response(audio, impulse_response):
	input_max = np.max(np.abs(audio))
	if len(impulse_response.shape) == 1: # mono impulse response
		if len(audio.shape) == 1: # mono input
			audio = np.convolve(audio, impulse_response)
		else:
			audio = np.array([np.convolve(audio[:, i], impulse_response) for i in range(2)]).T
	else:
		if len(audio.shape) == 1: # mono input
			audio = np.array([np.convolve(audio, impulse_response[:, i]) for i in range(2)]).T
		else:
			audio = np.array([np.convolve(audio[:, i], impulse_response[:, i]) for i in range(2)]).T
	return audio / np.max(np.abs(audio)) * input_max
