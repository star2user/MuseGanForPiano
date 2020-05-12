from src.MuseGAN import MuseGAN
import os
import matplotlib.pyplot as plt
import numpy as np
import types

from music21 import midi
from music21 import note, stream, duration

SECTION = 'compose'
RUN_ID = '0017'
DATA_NAME = 'chorales'

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
n_tracks = 2
z_dim = 32

x_train = np.load('easyScore2.npy')
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

gan.barGen[0].summary()
gan.generator.summary()
gan.critic.summary()

EPOCHS = 2000
PRINT_EVERY_N_BATCHES = 20

'''
gan.epoch = 0

'''

gan.load_weights(RUN_FOLDER)
gan.epoch = 2000;
'''
gan.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
)
'''
r = 5
chords_noise = np.random.normal(0, 1, (r, z_dim))
style_noise = np.random.normal(0, 1, (r, z_dim))
melody_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))
groove_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))
score = gan.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

'''
def TrimScore(score, leastNoteBeat):
    # score: (batchSize, 2, 96, 84 2) 형태의 배열
    # leastNoteBeat : 악보에서 나올 수 있는 음표의 최소박자
    output = np.array(score)

    batchSize = score.shape[0]
    count = score.shape[2] // leastNoteBeat;
    barCount = score.shape[1]
    pitchCount = score.shape[3]
    trackCount = score.shape[4]

    for dataNumber in range(batchSize):
      for trackNumber in range (trackCount):
          for barNumber in range (barCount):
             for i in range (leastNoteBeat):
                  for pitchNumber in range (pitchCount):
                      output[dataNumber, barNumber, i*count:(i+1)*count, pitchNumber, trackNumber] = score[dataNumber, barNumber, i*count, pitchNumber, trackNumber]

    return output
 '''
gan.notes_to_midi(RUN_FOLDER, score, 'Originalsample')
origninalScore = score

score = gan.TrimScore(origninalScore, 16)

gan.notes_to_midi(RUN_FOLDER, score, 'Trim16sample0')


