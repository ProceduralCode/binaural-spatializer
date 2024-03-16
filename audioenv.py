
import numpy as np
import audiolib as al

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
	'''An audio environment that can be used to apply spatialization to audio.'''
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
	binaural_irs = [al.read(f'hrirs/{dir}.wav') for dir in binarual_dir_info]
	max_ir_len = np.max([len(ir) for ir in binaural_irs])

	def __init__(self, ray_dists):
		'''ray_dists is a list of distances to the surrounding environment, in meters.'''
		self.ray_dists = ray_dists

	def get_echo(self, audio):
		# TODO: since this only has a single value for each ray, the convolution might be able to be optimized
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound) + 1))
		for ray_dist in self.ray_dists:
			ray_dist = max(ray_dist, 1)
			delay = int(ray_dist * speed_of_sound)
			impulse_response[delay] = 1 / ray_dist
		return al.apply_impulse_response(audio, impulse_response)

	def get_binaural_echo(self, audio):
		max_dist = np.max(self.ray_dists)
		impulse_response = np.zeros((int(max_dist * speed_of_sound) + self.max_ir_len, 2))
		for ray_dist, ir in zip(self.ray_dists, self.binaural_irs):
			ray_dist = max(ray_dist, 1)
			delay = int(ray_dist * speed_of_sound)
			impulse_response[delay:delay + len(ir)] += ir / ray_dist
		return al.apply_impulse_response(audio, impulse_response)

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
			irs.append(al.pad_to_length(ir * interp_weights[i], max_ir_len))
		return al.apply_impulse_response(audio, np.sum(irs, axis=0))

	def apply_spatialization(audio, source_rel_pos, source_env, listener_env, has_LOS, echo_mult=0.2):
		dist = max(source_rel_pos.length(), 1)
		direct = AudioEnv.apply_binaural_angle(audio, source_rel_pos) / dist
		if not has_LOS:
			# apply lowpass filter if no line of sight (as high frequencies are more likely to be blocked by obstacles)
			direct = al.pass_filter(direct, 1000, type='lowpass')
		source_echo = echo_mult * source_env.get_echo(audio) / dist
		incoming = al.combine(direct, source_echo)

		listener_echo = echo_mult * listener_env.get_binaural_echo(incoming)
		audio = al.combine(incoming, listener_echo)
		# this delay conceptually happens before applying listener_env,
		#   but I believe it's commutative and faster to apply it after
		delay = int(dist / speed_of_sound * al.sample_rate)
		audio = np.concatenate((np.zeros((delay, 2)), audio))
		if np.max(np.abs(audio)) > 1:
			audio /= np.max(np.abs(audio))
		return audio
