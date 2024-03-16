
import numpy as np
import audiolib as al
from audioenv import Vec3, AudioEnv

print('Starting')

audio = al.read('eld.wav')
room_env = AudioEnv([np.random.rand() * 2 + 1 for _ in range(26)])
al.write('eld_room.wav',       AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, True))
al.write('eld_room_nolos.wav', AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, False))

cath_env = AudioEnv([np.random.rand() * 20 + 10 for _ in range(26)])
al.write('eld_cath.wav',       AudioEnv.apply_spatialization(audio, Vec3(0, 4, 0), cath_env, cath_env, True))
al.write('eld_cath_right.wav', AudioEnv.apply_spatialization(audio, Vec3(4, 1, 0.4), cath_env, cath_env, True))

# near_left_wall = AudioEnv([
# 	'back-down-left':     1.1
# 	'back-down':          1.5
# 	'back-down-right':
# 	'left-down':
# 	'down':
# 	'right-down':
# 	'forward-down-left':
# 	'forward-down':
# 	'forward-down-right':
# 	'back-left':
# 	'back':
# 	'back-right':
# 	'right':
# 	'left':
# 	'forward-left':
# 	'forward':
# 	'forward-right':
# 	'back-up-left':
# 	'back-up':
# 	'back-up-right':
# 	'left-up':
# 	'up':
# 	'right-up':
# 	'forward-up-left':
# 	'forward-up':
# 	'forward-up-right':
# ])

print('Done')
