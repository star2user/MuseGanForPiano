import numpy as np
from pypianoroll import Multitrack, Track
from src.MuseGAN import MuseGAN

BATCH_SIZE = 32
n_bars = 4
n_steps_per_bar = 96
n_pitches = 84
n_tracks = 1
z_dim = 32

x_train = np.load('Train4,4.npy')
#zeros = np.zeros_like(x_train)

#x_train = np.concatenate((x_train, zeros), axis=4)

print(x_train.shape)
x_train = x_train.reshape(-1, n_bars, n_steps_per_bar, n_pitches, n_tracks)

gan = MuseGAN(input_dim = (n_bars, n_steps_per_bar, n_pitches, n_tracks)
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

length = len(x_train)

gan.notes_to_midi(run_folder='midi', output=x_train[0:10], filename="TestMidi")
gan.notes_to_midi(run_folder='midi', output=x_train[length-10:length-1], filename="TestMidi2")