
import scipy.io.wavfile as wavfile
from scipy.signal import iirfilter, sosfilt
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

# ir = read_audio('room_ir.wav')
# input = read_audio('my_voice.wav')
# output = apply_impulse_response(input, ir)
# play_audio(output)
# write_audio('output.wav', output)








def mix_audio(audio1, audio2, ratio):
	return audio1 * ratio + audio2 * (1 - ratio)

def pass_filter(audio, cutoff, order=4, type='highpass'):
	sos = iirfilter(order, cutoff, btype=type, fs=sample_rate, output='sos')
	if len(audio.shape) == 2:
		return np.array([sosfilt(sos, audio[:, i]) for i in range(2)]).T
	return sosfilt(sos, audio)



def synth():
	freq = 440
	saw = np.linspace(0, 1, sample_rate) * freq % 2 - 1
	out = pass_filter(saw, 800, type='lowpass')
	return out

def stereo_unison(freq, detune, spread, voices):
	saws = []
	for i in range(voices):
		voice_freq = freq * (1 + detune * (i - (voices - 1) / 2))
		pan = spread * (i - (voices - 1) / 2) / (voices - 1)
		saw = np.linspace(0, 1, sample_rate) * voice_freq % 2 - 1
		pan_saw = np.array([saw * (1 - pan), saw * (1 + pan)]).T
		saws.append(pan_saw)
	return np.mean(saws, axis=0)

audio = stereo_unison(440, 0.01, 1, 3)
audio = pass_filter(audio, 800, type='lowpass')
play_audio(audio)

