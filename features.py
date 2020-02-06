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
    f6 = functions.kurtosis(functions.instantaneous_frequency(input_signal))
    f7 = functions.gmax(input_signal)
    f8 = functions.mean_of_squared(functions.instantaneous_cn_absolute(input_signal))
    f9 = functions.std_deviation(abs(functions.instantaneous_cn_absolute(input_signal)))
    f10 = functions.std_deviation(functions.instantaneous_cn_absolute(input_signal))
    f11 = functions.cum20(input_signal)
    f12 = functions.cum21(input_signal)
    f13 = functions.cum40(input_signal)
    f14 = functions.cum41(input_signal)
    f15 = functions.cum42(input_signal)
    f16 = functions.cum60(input_signal)
    f17 = functions.cum61(input_signal)
    f18 = functions.cum62(input_signal)
    f19 = functions.cum63(input_signal)
    f20 = functions.meanAbsolute(input_signal)
    f21 = functions.sqrtAmplitude(input_signal)
    f22 = functions.ratioIQ(input_signal)
    result = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22]
    return result

def plotFeatures(modulations, snrValues):
    #sns.set()    
    figure_folder = pathlib.Path("./figures/")
    
    with open('features.txt', 'r') as f:
        titles = f.readlines()

    dataFolder = pathlib.Path("./features.pickle")
    with open(dataFolder, 'rb') as handle:
        data = pickle.load(handle)
    
    meanFeatures = np.zeros((len(data), len(data[0]), len(data[0][0][0]))) #mod/snr/ft1...ftn
    #Calculates the mean of all features
    for mod in range(len(data)):
        print("Calculating mean features from {}".format(modulations[mod]))
        for snr in range(len(data[0])):
            for feature in range(len(data[0][0][0])): 
                aux = np.array([])
                for frame in range(len(data[0][0])):
                    aux = np.append(aux, data[mod][snr][frame][feature])
                meanFeatures[mod][snr][feature] = np.mean(aux)                

    for feature in range(len(meanFeatures[mod][snr])):
        print("Ploting feature {}".format(feature))
        plt.figure(dpi=300)
        plt.xlabel('SNR')
        plt.xticks(np.arange(len(snrValues)), [str(i) for i in snrValues], rotation=20)    
        plt.ylabel('Value')
        plt.title('{}'.format(titles[feature].rstrip()))        
        for mod in range(len(meanFeatures)):
            aux = np.array([])
            for snr in range(len(meanFeatures[mod])):
                aux = np.append(aux, meanFeatures[mod][snr][feature])
            plt.plot(aux, label=modulations[mod])
        plt.legend(loc='best')
        plt.savefig(join(figure_folder,'Feature {}'.format(feature) + ".png"), bbox_inches='tight', dpi=300)
    
    #TODO:check if features calculations is correct

if __name__ == "__main__":
    print("Check main file for features specifications!")
    modulations = ['8PSK', '16PSK', '16QAM', '256QAM', 'BPSK', 'OQPSK']
    snr = np.linspace(-14, 20, 18, dtype=int)
    plotFeatures(modulations, snr)