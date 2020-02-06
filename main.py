import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from os import listdir
from os.path import isfile, join
import features as ft
import functions
from multiprocessing import Process

dataFolder = pathlib.Path("../dataset/201801a_splitted/")
onlyFiles = [f.split(".")[0] for f in listdir(dataFolder) if isfile(join(dataFolder, f))]  # List all data files available
frames = 250  # Up to 4096
frameSize = 250  # Up to 1024
nFeatures = 22

modulations = ['8PSK', '16PSK', '16QAM', '256QAM', 'BPSK', 'OQPSK']
#modulations = ['16PSK', '16QAM', '64QAM', 'BPSK', 'OQPSK']
snr = np.linspace(-14, 20, 18, dtype=int)

def generateFeaturesData(modulations, snr):
    # Reads the pickles and stores in 'data' array in the following format:
    data = np.empty((len(modulations), len(snr), frames,
                    frameSize), dtype=np.complex)
    for idx, mod in enumerate(modulations):
        print("Extracting data from {}".format(mod))
        for idx_, snr_ in enumerate(snr):
            for f in onlyFiles:
                if f.split("_")[0] == mod and f.split("_")[1] == str(snr_):
                    with open(join(dataFolder, f + ".pickle"), 'rb') as handle:
                        signal_data = pickle.load(handle)
                        signal = np.empty((frames, frameSize), dtype=np.complex)
                        for frame in range(frames):
                            for sample in range(frameSize):
                                signal[frame][sample] = complex(
                                    signal_data[frame][sample][0], signal_data[frame][sample][1])
                        data[idx, idx_, :, :] = signal[:, :]
                        break

    #Calculate features and save it to a pickle file
    features = np.zeros((len(modulations), len(snr), frames, nFeatures))

    for mod in range((len(data))):
        print("Extracting features from {}".format(modulations[mod]))
        for snr in range(len(data[mod])):
            for frame in range(len(data[mod][snr])):
                features[mod][snr][frame][:] = ft.calculate_features(data[mod][snr][frame][:])

    with open('features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del features

generateFeaturesData(modulations, snr)
functions.convertToMat('features.pickle')
ft.plotFeatures(modulations, snr)

print('well done!')