
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

def pad_audio_to_length(audio, length):
	if len(audio) >= length:
		return audio
	if len(audio.shape) == 1:
		return np.pad(audio, (0, length - len(audio)))
	return np.pad(audio, ((0, length - len(audio)), (0, 0)))

def combine_audio(audio1, audio2):
	if len(audio1.shape) != len(audio2.shape):
		raise ValueError('Audio channels do not match')
	if audio1.shape[0] > audio2.shape[0]:
		audio2 = pad_audio_to_length(audio2, audio1.shape[0])
	else:
		audio1 = pad_audio_to_length(audio1, audio2.shape[0])
	return audio1 + audio2

def mix_audio(audio1, audio2, ratio):
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

class Vec3:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
	def __add__(self, other):
		return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
	def __sub__(self, other):
		return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
	def __mul__(self, scalar):
		return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
	def __truediv__(self, scalar):
		return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
	def length(self):
		return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
	def norm(self):
		return self / self.length()
	def dot(self, other):
		return self.x * other.x + self.y * other.y + self.z * other.z
	def cross(self, other):
		return Vec3(
			self.y * other.z - self.z * other.y,
			self.z * other.x - self.x * other.z,
			self.x * other.y - self.y * other.x
		)


speed_of_sound = 343 # m/s

class AudioEnv:
	binarual_dir_info = {
		'back-down-left':     Vec3( -1, -1, -1).norm(),
		'back-down':          Vec3(  0, -1, -1).norm(),
		'back-down-right':    Vec3(  1, -1, -1).norm(),
		'left-down':          Vec3( -1,  0, -1).norm(),
		'down':               Vec3(  0,  0, -1).norm(),
		'right-down':         Vec3(  1,  0, -1).norm(),
		'forward-down-left':  Vec3( -1,  1, -1).norm(),
		'forward-down':       Vec3(  0,  1, -1).norm(),
		'forward-down-right': Vec3(  1,  1, -1).norm(),
		'back-left':          Vec3( -1, -1,  0).norm(),
		'back':               Vec3(  0, -1,  0).norm(),
		'back-right':         Vec3(  1, -1,  0).norm(),
		'right':              Vec3(  1,  0,  0).norm(),
		'left':               Vec3( -1,  0,  0).norm(),
		'forward-left':       Vec3( -1,  1,  0).norm(),
		'forward':            Vec3(  0,  1,  0).norm(),
		'forward-right':      Vec3(  1,  1,  0).norm(),
		'back-up-left':       Vec3( -1, -1,  1).norm(),
		'back-up':            Vec3(  0, -1,  1).norm(),
		'back-up-right':      Vec3(  1, -1,  1).norm(),
		'left-up':            Vec3( -1,  0,  1).norm(),
		'up':                 Vec3(  0,  0,  1).norm(),
		'right-up':           Vec3(  1,  0,  1).norm(),
		'forward-up-left':    Vec3( -1,  1,  1).norm(),
		'forward-up':         Vec3(  0,  1,  1).norm(),
		'forward-up-right':   Vec3(  1,  1,  1).norm(),
	}
	binarual_dirs = list(binarual_dir_info.values())
	binaural_irs = [read_audio(f'birs/{dir}.wav') for dir in binarual_dir_info]
	max_ir_len = np.max([len(ir) for ir in binaural_irs])

	def __init__(self, ray_dists):
		self.ray_dists = ray_dists

	def get_echo(self, audio):
		# TODO: since this only has a single value for each ray, the convolution might be able to be optimized
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound) + 1))
		for ray_dist in self.ray_dists:
			ray_dist = max(ray_dist, 1)
			delay = int(ray_dist * speed_of_sound)
			impulse_response[delay] = 1 / ray_dist
		return apply_impulse_response(audio, impulse_response)

	def get_binaural_echo(self, audio):
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound) + self.max_ir_len, 2))
		for ray_dist, ir in zip(self.ray_dists, self.binaural_irs):
			ray_dist = max(ray_dist, 1)
			delay = int(ray_dist * speed_of_sound)
			impulse_response[delay:delay + len(ir)] += ir / ray_dist
		return apply_impulse_response(audio, impulse_response)

	def apply_binaural_angle(audio, source_rel_pos):
		if source_rel_pos.length() == 0:
			return audio
		source_dir = source_rel_pos.norm()

		# Find closest 3 dirs
		dists = [(source_dir - ray_dir).length() for ray_dir in AudioEnv.binarual_dirs]
		# closest_dirs = [self.ray_dirs[i] for i in np.argsort(dists)[:3]]
		closest_dirs_idxs = np.argsort(dists)[:3]
		closest_dirs = [AudioEnv.binarual_dirs[i] for i in closest_dirs_idxs]

		# Interpolate between the 3 closest rays using area percentages.
		#   Use the area of the triangle opposite of each ray as the weight.
		#   This makes the weight approach [1,0,0] as the source dir approaches the first ray
		#     and approaches [1/3,1/3,1/3] when the source dir is equidistant from all 3 rays.
		#   I don't believe this is the best way to interpolate, but it is likely good enough.
		def triangle_area(a, b, c):
			'''Area of the triangle with 3D vertices a, b, and c'''
			return (a - c).cross(b - c).length() / 2
		interp_weights = [
			triangle_area(source_dir, closest_dirs[1], closest_dirs[2]),
			triangle_area(source_dir, closest_dirs[0], closest_dirs[2]),
			triangle_area(source_dir, closest_dirs[0], closest_dirs[1])
		]
		interp_weights /= np.sum(interp_weights)    # sum(interp_weights) should now be 1

		max_ir_len = np.max([len(AudioEnv.binaural_irs[i]) for i in closest_dirs_idxs])
		irs = []
		for i, idx in enumerate(closest_dirs_idxs):
			ir = AudioEnv.binaural_irs[idx]
			irs.append(pad_audio_to_length(ir * interp_weights[i], max_ir_len))
		return apply_impulse_response(audio, np.sum(irs, axis=0))

	def apply_spatialization(audio, source_rel_pos, source_env, listener_env, has_LOS, echo_mult=0.2):
		dist = max(source_rel_pos.length(), 1)
		direct = AudioEnv.apply_binaural_angle(audio, source_rel_pos) / dist
		if not has_LOS:
			# apply lowpass filter if no line of sight (as high frequencies are more likely to be blocked by obstacles)
			direct = pass_filter(direct, 1000, type='lowpass')
		source_echo = echo_mult * source_env.get_echo(audio) / dist
		incoming = combine_audio(direct, source_echo)

		listener_echo = echo_mult * listener_env.get_binaural_echo(incoming)
		audio = combine_audio(incoming, listener_echo)
		# this delay conceptually happens before applying listener_env,
		#   but I believe it's commutative and faster to apply it after
		delay = int(dist / speed_of_sound * sample_rate)
		audio = np.concatenate((np.zeros((delay, 2)), audio))
		if np.max(np.abs(audio)) > 1:
			audio /= np.max(np.abs(audio))
		return audio




print('Starting')

audio = read_audio('elderberries.wav')
room_env = AudioEnv([np.random.rand() * 2 + 1 for _ in range(26)])
write_audio('eld_room.wav',       AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, True))
write_audio('eld_room_nolos.wav', AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, False))

cath_env = AudioEnv([np.random.rand() * 20 + 10 for _ in range(26)])
write_audio('eld_cath.wav', AudioEnv.apply_spatialization(audio, Vec3(0, 4, 0), cath_env, cath_env, True))
write_audio('eld_cath_right.wav', AudioEnv.apply_spatialization(audio, Vec3(4, 1, 0.4), cath_env, cath_env, True))

print('Done')


























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

# audio = stereo_unison(440, 0.01, 1, 3)
# audio = pass_filter(audio, 800, type='lowpass')
# play_audio(audio)






