# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import png
from PIL import Image
import numpy as np
import noisereduce as nr
import librosa as lb
import cv2
import librosa.display as lbd
import soundfile as sf
from pip._internal.commands import install
from soundfile import SoundFile
import pandas as pd
from IPython.display import Audio
from python_speech_features import mfcc
from pathlib import Path
import time

import timm
import tensorflow as tf

from matplotlib import pyplot as plt

import reverse_geocode
import os, random, gc
import pycountry_convert as pc
import re, time, json
from ast import literal_eval

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import joblib
from fnmatch import fnmatch

import geopandas as gpd

import plotly_express as px

# Read in the raw metadata
train_meta = pd.read_csv('/Users/matthewrusso/train_metadata.csv',
                         usecols=['primary_label', 'latitude', 'longitude', 'common_name',
                                  'time', 'filename', 'rating'])

# This will get you all of the bird names that we have files for on the computer
base_path = "/Users/matthewrusso/train_short_audio/"
# birds=[] # list of all birds
# for root, dirs, files in os.walk(base_path):
#    if root == base_path: # matches first element
#        birds = dirs # returns first element
# path_audio = "acafly/"

# This will give you all of the names of birds in the US that we have files for on the computer
coord_pairs = np.column_stack((train_meta['latitude'].values,
                               train_meta['longitude'].values))
loc_from_coords = reverse_geocode.search(coord_pairs)
countries = np.array([d['country'] for d in loc_from_coords])
# country_alpha2s = np.array([pc.country_name_to_country_alpha2(country) for country in countries])
# continents = np.array([pc.country_alpha2_to_continent_code(country_alpha2) for country_alpha2 in country_alpha2s])
# print(set(continents))
# US_indices = np.where(countries == 'United States')
USA = countries == 'United States'
CAN = countries == 'Canada'
MEX = countries == 'Mexico'
NA = (USA | CAN | MEX)
NA_ind = np.where(NA)
train_meta_NA = train_meta.iloc[NA_ind[0]]
min_rating = 3
train_meta_NA_by_rating = train_meta_NA[train_meta_NA["rating"] >= min_rating]
#train_meta_US_by_rating.groupby('primary_label').size()
train_meta_NA_sort = train_meta_NA_by_rating.groupby('primary_label').size().sort_values(ascending=False)
#train_meta_US_final = train_meta_US_sort[:train_meta_US_sort.median().astype(int)]
# print(train_meta_US_sort.index)

birds50 = []
flist = []  # list of all files
blist = []  # list of files for one bird
i50 = 0

for i, bird in enumerate(train_meta_NA_sort.index):
    cond = train_meta_NA_by_rating["primary_label"] == bird
    good_files = train_meta_NA_by_rating[cond].filename.values
    for root, dirs, files in os.walk(base_path + bird):
        for file in files:
            if any(file == audio_file for audio_file in good_files):
                blist.append(os.path.join(root, file))
    if len(blist) >= 100:
        i50 = i50 + 1
        print(i50, ". Found ", len(blist), ' files for ', bird, '(', i + 1, ')')
        birds50.append(bird)
        flist.append(blist)
    blist = []

print('Number of directories to process: ', len(flist))

# Here we do some pre-processing and create the 3-channel spectrograms for
# each audio file. This includes:
# 1. Find the stft of the time-series
# 2. Filter out the percussive noises
# 3. ...
def saveChannels(input_signal, start, end):
    gc.enable()
    # MK_spectrogram modified
    # Physical time for each column in stft is n_FFT/sr seconds (number of samples/sample rate)
    # Default: sr = 22050
    N_FFT = 1024  # ~23ms
    HOP_SIZE = N_FFT / 2  # number of samples between successive frames
    N_MELS = 256  # number of Mel bands to generate
    WIN_SIZE = N_FFT  # Equal to NFFT, so no zero-padding needed; ~23ms
    WINDOW_TYPE = 'hann'  # Hanning window
    FMIN = 1400  # lowest frequency (in Hz)

    y = lb.stft(input_signal[start:end], n_fft=2048,
                hop_length=862,
                win_length=2048,
                window="hann")
    harm = np.abs((lb.decompose.hpss(y, kernel_size=1, margin=1))[0]) ** 2
    # First Spectrogram: Window size = ~23ms, hop length = ~11.5ms
    mel_ch1 = lb.core.amplitude_to_db(lb.feature.melspectrogram(S=harm, sr=sr,
                                                                n_fft=N_FFT,
                                                                hop_length=HOP_SIZE,
                                                                n_mels=N_MELS,
                                                                htk=True,
                                                                fmin=FMIN,  # higher limit ##high-pass filter freq.
                                                                fmax=sr / 2))  # highest frequency (in Hz)
    # Second Spectrogram: Window size = ~46ms, hop length = ~23ms
    mel_ch2 = lb.core.amplitude_to_db(lb.feature.melspectrogram(S=harm, sr=sr,
                                                                n_fft=2 * N_FFT,
                                                                hop_length=2 * HOP_SIZE,
                                                                n_mels=N_MELS,
                                                                htk=True,
                                                                fmin=FMIN,  # higher limit ##high-pass filter freq.
                                                                fmax=sr / 2))  # highest frequency (in Hz)
    # Third Spectrogram: Window size = ~93ms, hop length = ~46ms
    mel_ch3 = lb.core.amplitude_to_db(lb.feature.melspectrogram(S=harm, sr=sr,
                                                                n_fft=4 * N_FFT,
                                                                hop_length=4 * HOP_SIZE,
                                                                n_mels=N_MELS,
                                                                htk=True,
                                                                fmin=FMIN,  # higher limit ##high-pass filter freq.
                                                                fmax=sr / 2))  # highest frequency (in Hz)
    return np.dstack((mel_ch1, mel_ch2, mel_ch3))

data = []
labels = []
for bird, birdList in enumerate(flist):
    print("Processing bird", bird, ':', birds50[bird], "...")
    j = 0
    tqdm_tot = min(200, len(birdList))
    for birdnr, path in tqdm(enumerate(birdList), total=tqdm_tot):
        cond_j_1 = j < 200
        cond_j_2 = j < len(birdList)
        if cond_j_1 and cond_j_2:
            # load the ogg file
            directory = base_path + str(bird) + birds50[bird][:5] + "/"
            # rsplit with '/' and 1 is taking the last part of path after /
            # if not os.path.exists(directory + path.rsplit('/', 1)[1].replace(' ', '')[:-4] + "_1.png"):
            input_signal, sr = lb.load(path, duration=30, mono=True)  # sr = sampling rate
            # Reduce the noise in the audio data before finding the spectrograms. The argument prop_decrease
            # determines to what extent to reduce noise - 1 for all, 0 for not at all
            input_signal = nr.reduce_noise(audio_clip=input_signal, noise_clip=input_signal, n_std_thresh=2, prop_decrease=1,
                                     verbose=False)
            # Determine the harmonic part of the de-noised audio file
            signal_harm = lb.effects.harmonic(input_signal)
            # Determine the onsets events in the de-noised audio file
            onsets = lb.onset.onset_detect(y=signal_harm, sr=sr, units='time')
            if not onsets.size == 0:
                start = int(onsets[0] * sr)
                end = int(start + 10*sr)  # 10 seconds
                melspecs = saveChannels(input_signal, start, end)
            else:
                melspecs = saveChannels(input_signal, 0, 10*sr)
            data.append(melspecs)
            labels.append(birds50[bird])
            pass
            j = j + 1
    print(len(data), data[0].shape)

bad_ind = []
for i in range(len(data)):
    if data[i].shape[1] != 256:
        bad_ind.append(i)
for index in sorted(bad_ind, reverse=True):
    del data[index]
    del labels[index]
data = np.stack(data)
labels = np.stack(labels)
np.save('feature_data_256', data)
np.save('feature_labels_256', labels)
# print(train_meta.head())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
