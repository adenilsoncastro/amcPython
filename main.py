import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import features as ft
import pathlib
from os import listdir
from os.path import isfile, join

dataFolder = pathlib.Path("../dataset/201801a_splitted/")
onlyFiles = [f.split(".")[0] for f in listdir(dataFolder) if isfile(join(dataFolder, f))]  # List all data files available
frameSize = 1024  # Up to 1024
frames = 100  # Up to 4096
nFeatures = 9

modulations = ['16QAM', 'BPSK']
snr = ['0']

# Reads the pickles and stores in 'data' array in the following format:
data = np.empty((len(modulations), len(snr), frames,
                 frameSize), dtype=np.complex)
for idx, mod in enumerate(modulations):
    print("Extracting data from {}".format(mod))
    temp = np.array([])
    for idx_, snr_ in enumerate(snr):
        for f in onlyFiles:
            if f.split("_")[0] == mod and f.split("_")[1] == snr_:
                with open(join(dataFolder, f + ".pickle"), 'rb') as handle:
                    signal_data = pickle.load(handle)
                    signal = np.empty((frames, frameSize), dtype=np.complex)
                    for frame in range(frames):
                        for sample in range(frameSize):
                            signal[frame][sample] = complex(
                                signal_data[frame][sample][0], signal_data[frame][sample][1])
                    data[idx, idx_, :, :] = signal[:, :]
                    break

plt.plot(data[0][0][0][:].real, color='blue')
plt.plot(data[0][0][0][:].imag, color='green')
plt.title('i/q data')
plt.show()

features = np.zeros((frames, nFeatures))
for idx in range(nFeatures):
    features[idx][:] = ft.calculate_features(data[0][0][0][:])

print('well done!')