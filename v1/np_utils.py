import numpy as np
# Generating the current and the previous track

# np_dir = '/home/ashar/Documents/ece6254/project/data/'

np_dir = '/home/ashar/Documents/ece6254/project/data/musegan/'

midiset = np.load(np_dir + 'tra_phr.npy')
counter = 0

truncated_midi100 = midiset
prev_truncated_midi100 = np.zeros(truncated_midi100.shape)

prev_truncated_midi100[:, 1:100, ...] = truncated_midi100[:, 0:99, ...]

truncated_midi100 = truncated_midi100.reshape((-1, 16, 128, 1))
prev_truncated_midi100 = prev_truncated_midi100.reshape((-1, 16, 128, 1))


print truncated_midi100.shape
print prev_truncated_midi100.shape
np.save(np_dir + 'truncated_midi100.npy', truncated_midi100)
np.save(np_dir + 'prev_truncated_midi100.npy', prev_truncated_midi100)


#
# truncated_midi = np.zeros((333, 80, 16, 128, 1))
# prev_truncated_midi = np.zeros((333, 80, 16, 128, 1))
# for idx in range(midiset.shape[0]):
#     temp_data = midiset[idx]
#     if temp_data.shape[0] > 80:
#         truncated_midi[counter, ..., 0] = temp_data[0:80, ...]
#         prev_truncated_midi[counter,  1:80, ..., 0] = temp_data[1:80, ...]
#         counter += 1
#
# truncated_midi = truncated_midi.reshape((-1, 16, 128, 1))
# prev_truncated_midi = prev_truncated_midi.reshape((-1, 16, 128, 1))
#
# print truncated_midi.shape
# print prev_truncated_midi.shape
# np.save(np_dir + 'truncated_midi.npy', truncated_midi)
# np.save(np_dir + 'prev_truncated_midi.npy', prev_truncated_midi)
