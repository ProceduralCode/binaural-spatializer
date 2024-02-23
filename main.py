
import scipy.io.wavfile as wavfile
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

sample_rate = 48000

def read_audio(filename):
	global sample_rate
	sample_rate, audio = wavfile.read(filename)
	if audio.dtype.kind == 'f':
		return audio
	return audio / np.iinfo(audio.dtype).max    # normalize to [-1, 1]

def write_audio(filename, audio):
	data = (audio * np.iinfo(np.int16).max).astype(np.int16)
	wavfile.write(filename, sample_rate, data)

def play_audio(audio):
	data = (audio * np.iinfo(np.int16).max).astype(np.int16)
	sd.play(data, sample_rate)
	sd.wait()

def display_audio(audio):
	plt.plot(audio)
	plt.show()

def apply_impulse_response(audio, impulse_response):
	input_max = np.max(np.abs(audio))
	if len(impulse_response.shape) == 2:
		audio = np.array([np.convolve(audio[:, i], impulse_response[:, i]) for i in range(2)]).T
	else:
		audio = np.convolve(audio, impulse_response)
	return audio / np.max(np.abs(audio)) * input_max

ir = read_audio('room_ir.wav')
input = read_audio('my_voice.wav')
output = apply_impulse_response(input, ir)
play_audio(output)
write_audio('output.wav', output)

