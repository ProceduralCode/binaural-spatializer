
import numpy as np
import audiolib as al
from audioenv import Vec3, AudioEnv

print('Starting')

audio = al.read('audio/eld.wav')

room_env = AudioEnv([np.random.rand() * 2 + 1 for _ in range(26)])
al.write('audio/eld_room.wav',       AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, True))
al.write('audio/eld_room_nolos.wav', AudioEnv.apply_spatialization(audio, Vec3(0, 1, 0), room_env, room_env, False))

cath_env = AudioEnv([np.random.rand() * 20 + 10 for _ in range(26)])
al.write('audio/eld_cath.wav',       AudioEnv.apply_spatialization(audio, Vec3(0, 4, 0), cath_env, cath_env, True))
al.write('audio/eld_cath_right.wav', AudioEnv.apply_spatialization(audio, Vec3(4, 1, 0.4), cath_env, cath_env, True))

near_left_wall = AudioEnv([
	0.3,  # back-down-left
	1,    # back-down
	1.3,  # back-down-right
	0.2,  # left-down
	0.9,  # down
	1.2,  # right-down
	0.3,  # forward-down-left
	1,    # forward-down
	1.3,  # forward-down-right
	0.2,  # back-left
	4,    # back
	5.5,  # back-right
	6,    # right
	0.1,  # left
	0.2,  # forward-left
	1,    # forward
	2.5,  # forward-right
	0.3,  # back-up-left
	1.5,  # back-up
	2,    # back-up-right
	0.2,  # left-up
	0.8,  # up
	1.4,  # right-up
	0.3,  # forward-up-left
	1.4,  # forward-up
	1.6,  # forward-up-right
])
room_center = AudioEnv([3 for _ in range(26)])
al.write('audio/eld_near_left_wall.wav', AudioEnv.apply_spatialization(audio, Vec3(3, 0, 0), room_center, near_left_wall, True))

print('Done')
