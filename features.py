import functions
import pathlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import join

def calculate_features(input_signal):
    f1 = functions.std_deviation((abs(functions.instantaneous_phase(input_signal))))
    f2 = functions.std_deviation(functions.instantaneous_phase(input_signal))
    f3 = functions.std_deviation((abs(functions.instantaneous_frequency(input_signal))))
    f4 = functions.std_deviation(functions.instantaneous_frequency(input_signal))
    f5 = functions.kurtosis(functions.instantaneous_absolute(input_signal))
    f6 = functions.gmax(input_signal)
    f7 = functions.mean_of_squared(functions.instantaneous_cn_absolute(input_signal))
    f8 = functions.std_deviation(abs(functions.instantaneous_cn_absolute(input_signal)))
    f9 = functions.std_deviation(functions.instantaneous_cn_absolute(input_signal))
    result = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    return result

def plotFeatures():
    #sns.set()
    modulations = ['8PSK', '16PSK', '16QAM', '64QAM', '256QAM', 'BPSK', 'QPSK'] #TODO: receive these as functions parameters
    snr = np.linspace(-14, 20, 18, dtype=int)
    figure_folder = pathlib.Path("./figures/")

    dataFolder = pathlib.Path("./features.pickle")
    with open(dataFolder, 'rb') as handle:
        data = pickle.load(handle)
    
    meanFeatures = np.zeros((len(data), len(data[0]), len(data[0][0][0]))) #mod/snr/ft1...ftn
    #Calculates the mean of all features
    for mod in range(len(data)):
        for snr in range(len(data[0])):
            for feature in range(len(data[0][0][0])): 
                aux = np.array([])
                for frame in range(len(data[0][0])):
                    aux = np.append(aux, data[mod][snr][frame][feature])
                meanFeatures[mod][snr][feature] = np.mean(aux)    

    for feature in range(len(meanFeatures[mod][snr])):
        plt.figure(figsize=(6.4, 3.6), dpi=300)
        plt.xlabel('SNR')
        plt.ylabel('Value')
        plt.title('Feature {}'.format(feature))        
        for mod in range(len(meanFeatures)):
            aux = np.array([])
            for snr in range(len(meanFeatures[mod])):
                aux = np.append(aux, meanFeatures[mod][snr][feature])
            plt.plot(aux, label=modulations[mod])   
            #df = pd.DataFrame(aux)
            #sns.relplot(data=df, kind="line")
        plt.legend(loc='best')
        plt.savefig(join(figure_folder,'Feature {}'.format(feature) + ".png"), bbox_inches='tight', dpi=300)
    
    #TODO:check if features calculations is correct
if __name__ == "__main__":
    print("Check main file for features specifications!")
    plotFeatures()