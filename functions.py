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
    aux= input ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return abs(output)

def cum21(input):
    aux = abs(input) ** 2
    aux1 = sum(aux)
    aux2 = 1/len(input)
    output = aux2 * aux1
    return output

def cum40(input):
    aux = input ** 4
    aux2 = cum20(input)
    aux3 = aux2 ** 2
    aux4 = 3 * aux3
    aux5 = aux - aux4
    aux6 = sum(aux5)
    aux7 = 1/len(input)
    output = aux7 * aux6
    return abs(output)

def cum41(input):
    c20 = cum20(input)
    c21 = cum21(input)
    aux = 3 * c20 * c21
    aux1 = input ** 3
    aux2 = np.conj(input)
    aux3 = aux1 * aux2
    aux4 = aux3 - aux
    aux5 = sum(aux4)
    aux6 = 1/len(input)
    output = aux5 * aux6
    return abs(output)

def cum42(input):
    c20 = abs(cum20(input)) ** 2
    aux = cum21(input) ** 2
    c21 = 2 * aux
    aux1 = abs(input) ** 4
    aux2 = aux1 - c20 - c21
    aux3 = sum(aux2)
    aux4 = 1/len(input)
    output = aux3 * aux4
    return output

def cum60(input):    
    input_6th = input ** 6
    input_4th = input ** 4
    cum_rd = cum20(input) ** 3
    
    term2 = 15 * cum20(input) * input_4th
    term3 = 3 * cum_rd
    
    aux = sum(input_6th - term2 + term3)
    aux2 = 1/len(input)
    output = aux2 * aux
    return abs(output)

def cum61(input):
    input_conj = np.conj(input)
    input_rd = input ** 3
    input_4th = input ** 4
    input_5th = input ** 5
    cum_nd = cum20(input) ** 2

    term1 = input_5th * input_conj
    term2 = 5 * cum21(input) * input_4th
    term3 = 10 * cum20(input) * input_rd * input_conj
    term4 = 30 * cum_nd * cum21(input)
    
    aux = sum(term1 - term2 - term3 + term4)
    aux2 = 1/len(input)
    output = aux2 * aux
    return abs(output)

def cum62(input):    
    input_nd = input ** 2
    input_rd = input ** 3
    input_4th = input ** 4
    input_conj = np.conj(input)
    input_conj_nd = np.conj(input) ** 2
    cum20_nd = cum20(input) ** 2
    cum21_nd = cum21(input) ** 2

    term1 = input_4th * input_conj_nd
    term2 = 6 * cum20(input) * input_nd * input_conj_nd
    term3 = 8 * cum21(input) * input_rd * input_conj
    term4 = input_conj_nd * input_4th
    term5 = 6 * cum20_nd * input_conj_nd
    term6 = 24 * cum21_nd * cum20(input)
    
    aux = sum(term1 - term2 - term3 - term4 + term5 + term6)
    aux2 = 1/len(input)
    output = aux2 * aux
    return abs(output)

def cum63(input):
    input_nd = input ** 2
    input_rd = input ** 3
    input_conj = np.conj(input)
    input_conj_nd = np.conj(input) ** 2
    input_conj_rd = np.conj(input) ** 3
    cum21_rd = cum21(input) ** 3

    term1 = input_rd * input_conj_rd
    term2 = 9 * cum21(input) * input_nd * input_conj_nd
    term3 = 12 * cum21_rd
    term4 = 3 * cum20(input) * input * input_conj_rd
    term5 = 3 * input_conj_nd * input_rd * input_conj
    term6 = 18 * cum20(input) * cum21(input) * input_conj_nd
    
    aux = sum(term1 - term2 + term3 - term4 - term5 + term6)
    aux2 = 1/len(input)
    output = aux2 * aux
    return abs(output)

#TODO: check new features calculation

def convertToMat(data):    
    matFolder = pathlib.Path("./Matlab/")

    try:
        with open(data, 'rb') as handle:
            aux = pickle.load(handle)
            scipy.io.savemat(join(matFolder, "features.mat"), mdict={'pickle_data':aux})
    except:
        print("An error has occurred throughout files conversion!")
