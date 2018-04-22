#These scripts are refer to "https://github.com/carpedm20/DCGAN-tensorflow"
import os
import scipy.misc
import numpy as np
from model import MidiNet
from utils import pp, to_json, generation_test
from npy2midi_converter import write_piano_rolls_to_midi, set_piano_roll_to_instrument

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [20]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 72, "The size of batch [72]")
flags.DEFINE_integer("genre", 1, "The size of batch [72]")
flags.DEFINE_integer("num_program", 0, "The size of batch [72]")
flags.DEFINE_integer("output_w", 16, "The size of the output segs to produce [16]")
flags.DEFINE_integer("output_h", 128, "The size of the output note to produce [128]")
flags.DEFINE_integer("c_dim", 1, "Number of Midi track. [1]")
flags.DEFINE_integer("num_batches", 200, "Number of batches conctenated in a midi file. [1]")
flags.DEFINE_string("checkpoint_dir", "/media/ashar/Data/lmd_genre/lpd_5/midinet_ckpts_per_epoch/",
                    "Directory for [checkpoint]")

flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("dataset", "MidiNet_v1", "The name of dataset ")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("generation_test", False, "True for generation_test, False for nothing [False]")
flags.DEFINE_string("gen_dir", "gen", "Directory name to save the generate samples [samples]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.gen_dir):
        os.makedirs(FLAGS.gen_dir)

    with tf.Session() as sess:
        if FLAGS.dataset == 'MidiNet_v1':
            model = MidiNet(sess,  batch_size=FLAGS.batch_size, y_dim=4, output_w=FLAGS.output_w,
                            output_h=FLAGS.output_h, c_dim=FLAGS.c_dim, dataset_name=FLAGS.dataset,
                            is_crop=FLAGS.is_crop,  checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir,
                            gen_dir=FLAGS.gen_dir)

            model.load(FLAGS.checkpoint_dir)


            #For a song, we need to provide 100 batches of size 72, which would be concatenated to form a single song vector
            song_array = np.array([])
            genre_vec = np.zeros((FLAGS.batch_size, 4))

            # One hot for the genre
            genre_vec[:, 0:4] = [0, 0, 1, 0]

            for i in range(FLAGS.num_batches):
                batch_sample = generation_test(sess, model, FLAGS, genre_vec, option=0)
                song_array = np.concatenate([song_array, batch_sample], axis=0) if song_array.size else batch_sample

            song_array[np.isnan(song_array)] = 0
            song_array[song_array >= 0.8] = 1
            song_array[song_array < 0.8] = 0

            piano_roll = np.expand_dims(song_array.squeeze(), axis=0)
            output_path = os.path.join(FLAGS.gen_dir, str(FLAGS.genre))
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            write_piano_rolls_to_midi(piano_roll, program_nums=[FLAGS.num_program], is_drum=[False],
                                      filename=output_path + '/test2.mid',
                                      velocity=70, tempo=1000.0, beat_resolution=24)


if __name__ == '__main__':
    tf.app.run()