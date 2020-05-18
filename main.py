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
n_bars = 4
n_steps_per_bar = 96
n_pitches = 84
n_tracks = 1
z_dim = 32

x_train = np.load('Train4,4.npy')
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


EPOCHS = 4000
PRINT_EVERY_N_BATCHES = 100

gan.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
)


