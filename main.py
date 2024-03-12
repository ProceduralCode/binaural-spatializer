
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
	if len(impulse_response.shape) == 2: # stereo impulse response
		if len(audio.shape) == 1: # mono input
			audio = np.array([audio, audio]).T
		audio = np.array([np.convolve(audio[:, i], impulse_response[:, i]) for i in range(2)]).T
	else:
		audio = np.convolve(audio, impulse_response)
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
	def normalize(self):
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
	bounce_mult = 0.2 # how much sound is reflected off walls

	s2 = np.sqrt(2) / 2
	s3 = np.sqrt(3) / 3
	ray_dirs = [
		# XY plane
		Vec3(   1,   0,   0), # right
		Vec3(  s2,  s2,   0), # forward-right
		Vec3(   0,   1,   0), # forward
		Vec3( -s2,  s2,   0), # forward-left
		Vec3(  -1,   0,   0), # left
		Vec3( -s2, -s2,   0), # back-left
		Vec3(   0,  -1,   0), # back
		Vec3(  s2, -s2,   0), # back-right

		# XZ plane
		#                     # right
		Vec3(  s2,   0,  s2), # right-up
		Vec3(   0,   0,   1), # up
		Vec3( -s2,   0,  s2), # left-up
		#                     # left
		Vec3( -s2,   0, -s2), # left-down
		Vec3(   0,   0,  -1), # down
		Vec3(  s2,   0, -s2), # right-down

		# YZ plane
		# 						    # forward
		Vec3(   0,  s2,  s2), # forward-up
		# 						    # up
		Vec3(   0, -s2,  s2), # back-up
		# 						    # back
		Vec3(   0, -s2, -s2), # back-down
		# 						    # down
		Vec3(   0,  s2, -s2), # forward-down

		# 3D diagonals
		Vec3(  s3,  s3,  s3), # forward-up-right
		Vec3( -s3,  s3,  s3), # forward-up-left
		Vec3( -s3,  s3, -s3), # forward-down-left
		Vec3(  s3,  s3, -s3), # forward-down-right
		Vec3(  s3, -s3,  s3), # back-up-right
		Vec3( -s3, -s3,  s3), # back-up-left
		Vec3( -s3, -s3, -s3), # back-down-left
		Vec3(  s3, -s3, -s3)  # back-down-right
	]

	binaural_irs = [
		read_audio('right.wav'),
		read_audio('forward-right.wav'),
		read_audio('forward.wav'),
		read_audio('forward-left.wav'),
		read_audio('left.wav'),
		read_audio('back-left.wav'),
		read_audio('back.wav'),
		read_audio('back-right.wav'),
		read_audio('right-up.wav'),
		read_audio('up.wav'),
		read_audio('left-up.wav'),
		read_audio('left-down.wav'),
		read_audio('down.wav'),
		read_audio('right-down.wav'),
		read_audio('forward-up.wav'),
		read_audio('back-up.wav'),
		read_audio('back-down.wav'),
		read_audio('forward-down.wav'),
		read_audio('forward-up-right.wav'),
		read_audio('forward-up-left.wav'),
		read_audio('forward-down-left.wav'),
		read_audio('forward-down-right.wav'),
		read_audio('back-up-right.wav'),
		read_audio('back-up-left.wav'),
		read_audio('back-down-left.wav'),
		read_audio('back-down-right.wav')
	]
	max_ir_len = np.max([len(ir) for ir in binaural_irs])

	def __init__(self, ray_dists):
		self.ray_dists = ray_dists

	# TODO: since this only has a single value for each ray, the convolution might be able to be optimized
	def get_echo(self, audio):
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound)))
		for ray_dist in self.ray_dists:
			impulse_response[int(ray_dist * speed_of_sound)] = self.bounce_mult / ray_dist ** 2
		return apply_impulse_response(audio, impulse_response)

	def get_binaural_echo(self, audio):
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound) + self.max_ir_len))
		for ray_dist, ir in zip(self.ray_dists, self.binaural_irs):
			delay = int(ray_dist * speed_of_sound)
			impulse_response[delay:delay + len(ir)] += ir * self.bounce_mult / ray_dist ** 2
		return apply_impulse_response(audio, impulse_response)

	def apply_binaural_angle(self, audio, source_rel_pos):
		source_dir = source_rel_pos.normalized()

		# Find closest 3 dirs
		dists = [(source_dir - ray_dir).length() for ray_dir in self.ray_dirs]
		# closest_dirs = [self.ray_dirs[i] for i in np.argsort(dists)[:3]]
		closest_dirs_idxs = np.argsort(dists)[:3]
		closest_dirs = [self.ray_dirs[i] for i in closest_dirs_idxs]

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

		ir = np.sum([
			self.binaural_irs[closest_dirs_idxs[i]] * interp_weights[i] for i in range(3)
		], axis=0)
		return apply_impulse_response(audio, ir)





def apply_spatialization(audio, source_rel_pos, source_env, listener_env, has_LOS):
	echo = source_env.get_echo(audio)
	if not has_LOS:
		# apply lowpass filter if no LOS (as high frequencies are more likely to be blocked by obstacles)
		audio = pass_filter(audio, 1000, type='lowpass')
	dist = source_rel_pos.length()
	audio = (audio + echo) / dist ** 2
	audio = listener_env.get_binaural_echo(audio, source_rel_pos)
	# this delay conceptually happens before applying listener_env,
	#   but I believe it's commutative and faster to apply it after
	delay = int(dist * speed_of_sound)
	audio = np.concatenate((np.zeros(delay), audio))





# ir = read_audio('room_ir.wav')
ir = read_audio('left binaural impulse.wav')
input = read_audio('my_voice.wav')
output = apply_impulse_response(input, ir)
play_audio(output)
write_audio('output.wav', output)























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

# audio = stereo_unison(440, 0.01, 1, 3)
# audio = pass_filter(audio, 800, type='lowpass')
# play_audio(audio)






