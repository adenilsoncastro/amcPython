import math
import numpy as np
import scipy.io
import pathlib
from os.path import join
import pickle

def gmax(input):
    fftsquared = abs(np.fft.fft(input)) ** 2
    psd = fftsquared / len(input)
    output = max(psd)
    return output

def mean(input):
    output = sum(input) / len(input)
    return output

def mean_of_squared(input):
    aux1 = input ** 2
    aux2 = sum(aux1)
    output = aux2 / len(input)
    return output

def std_deviation(input):
    aux1 = (input - mean(input))
    aux2 = aux1 ** 2
    aux3 = sum(aux2)
    aux4 = 1 / (len(input) - 1)
    output = math.sqrt(aux3 * aux4)
    return output

def kurtosis(input):
    m = mean(input)
    aux4 = (input - m) ** 4
    aux2 = (input - m) ** 2
    num = (1 / len(input)) * sum(aux4)
    den = ((1 / len(input)) * sum(aux2)) ** 2
    output = num / den
    return output

def instantaneous_phase(input):
    output = np.angle(input)
    return output

def instantaneous_frequency(input):
    output = 1 / (2 * np.pi) * np.diff(np.unwrap(np.angle(input)))
    return output

def instantaneous_absolute(input):
    output = abs(input)
    return output

def instantaneous_cn_absolute(input):
    output = abs(input) / mean(abs(input)) - 1
    return output

def symmetry(input):
    
    return 1

def cum20(input):
    aux= abs(input) ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return output

def cum21(input):
    aux = input ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return output

def cum40(input):
    aux = input ** 4
    aux2 = cum20(input) ** 2
    aux3 = 3 * aux2
    aux4 = aux - aux3
    aux5 = sum(aux4)
    aux6 = 1/len(input)
    output = aux6 * aux5
    return output

#TODO: check new features calculation

def convertToMat(data):    
    matFolder = pathlib.Path("./Matlab/")

    try:
        with open(data, 'rb') as handle:
            aux = pickle.load(handle)
            scipy.io.savemat(join(matFolder, "features.mat"), mdict={'pickle_data':aux})
    except:
        print("An error has occurred throughout files conversion!")
