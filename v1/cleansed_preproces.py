import os
import numpy as np


INPUT_BASE_DIR = '/media/ashar/Data/lmd_genre/lpd_5/numpy_midi/'
genre_binarylist = [00, 01, 10, 11]


def data_loader(INPUT_BASE_DIR, genre_binarylist):
    full_X = np.array([])
    full_prev_X = np.array([])
    full_label_set = np.array([])

    for genre in genre_binarylist:
        final_out_path = os.path.join(INPUT_BASE_DIR, str(genre))
        if not os.path.exists(final_out_path):
            os.makedirs(final_out_path)
        label_arr = np.load(final_out_path + '/genre_label.npy')
        npy_arr = np.load(final_out_path + '/midi_bars.npy')
        prev_npy = np.zeros(npy_arr.shape)
        prev_npy[1:, ...] = npy_arr[:-1, ...]

        full_X = np.concatenate([full_X, npy_arr], axis = 0) if full_X.size else npy_arr
        full_prev_X = np.concatenate([full_prev_X, prev_npy], axis=0) if full_prev_X.size else prev_npy
        full_label_set = np.concatenate([full_label_set, label_arr], axis=0) if full_label_set.size else label_arr

    return full_X, full_prev_X, full_label_set

check = 1
    # prev_npy[]


