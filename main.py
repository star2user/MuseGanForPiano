from src.MuseGAN import MuseGAN
import os
import matplotlib.pyplot as plt
import numpy as np
import types

from music21 import midi
from music21 import note, stream, duration

RUN_FOLDER = 'midi/'

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'samples'))
#load data
BATCH_SIZE = 32
n_bars = 2
n_steps_per_bar = 96
n_pitches = 84
n_tracks = 1
z_dim = 32

x_train = np.load('trainingSet/classic_training.npy')
print(x_train.shape)
x_train = x_train.reshape(-1, n_bars, n_steps_per_bar, n_pitches, n_tracks)
print(x_train.shape)
print(x_train.shape[1:])

gan = MuseGAN(input_dim = x_train.shape[1:]
        , critic_learning_rate = 0.001
        , generator_learning_rate = 0.001
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = z_dim
        , batch_size = BATCH_SIZE
        , n_tracks = n_tracks
        , n_bars = n_bars
        , n_steps_per_bar = n_steps_per_bar
        , n_pitches = n_pitches
        )

gan.load_weights(RUN_FOLDER)

EPOCHS = 12000
PRINT_EVERY_N_BATCHES = 100
gan.epoch = 6000

gan.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
)


'''
r = 4

chords_noise = np.random.normal(0, 1, (r, z_dim))
style_noise = np.random.normal(0, 1, (r, z_dim))
melody_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))
groove_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))
gen_scores = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

gan.notes_to_midi(RUN_FOLDER, gen_scores, 'classicTrainingOriginal')
gan.notes_to_midi(RUN_FOLDER, gan.TrimScore(gen_scores, 16), '16th{}'.format('classicTraining'))
'''