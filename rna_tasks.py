import argparse
import json
import os
import pathlib
import pickle
import time
import uuid
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras import Sequential
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Loads JSON file with execution setup
with open("./info.json") as handle:
    info_json = json.load(handle)

# Config variables based on the JSON file
number_of_frames = info_json['numberOfFrames']
number_of_features = len(info_json['features']['using'])
number_of_snr = len(info_json['snr']['using'])
snr_list = info_json['snr']['using']
modulations = info_json['modulations']['names']

features_files = [f + "_features.pickle" for f in modulations]
test_data_files = [f + "_features.pickle" for f in modulations]

def quantize_rna(id):
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))

    print("Starting model quantization...")
    rna = join(str(rna_folder), "rna-" + id + ".h5")
    model = load_model(rna)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optmizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(join(rna_folder, 'rna-' + id + '_lite.h5'), 'wb') as f:
        f.write(tflite_model)

    print("Starting model evaluation...")
    result = np.zeros((len(modulations), number_of_snr))
    interpreter = tf.lite.Interpreter(model_path=join(rna_folder, 'rna-' + id + '_lite.h5'))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    for i, mod in enumerate(test_data_files):
        print("Evaluating {}".format(mod.split("_")[0]))
        
        with open(join(data_folder, mod), 'rb') as evaluating_data:
            data = pickle.load(evaluating_data)
        
        for j, snr in enumerate(snr_list):
            right_label = []
            predicted_label = []
            for _ in range(500):
                random_sample = np.random.choice(data[snr][:].shape[0], 1)
                data_test = [data[snr][i] for i in random_sample]
                data_test = normalize(data_test, norm='l2')                
                partial_right_label = info_json['modulations']['labels'][mod.split("_")[0]]
                data_test_float32 = data_test.astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], data_test_float32)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                partial_predicted_label = np.argmax(output_data)
                right_label.append(partial_right_label)
                predicted_label.append(partial_predicted_label)

            accuracy = accuracy_score(right_label, predicted_label)          
            result[i][j] = accuracy
    
    figure = plt.figure(figsize=(8, 4), dpi=150)
    plt.title("Accuracy - Lite model")
    plt.ylabel("Right prediction")
    plt.xlabel("SNR [dB]")
    plt.xticks(np.arange(number_of_snr), [info_json['snr']['values'][i] for i in info_json['snr']['using']])
    for item in range(len(result)):
        plt.plot(result[item], label=modulations[item])
    plt.legend(loc='best')
    plt.savefig(join(fig_folder, "accuracy_lite_" + id + ".png"), bbox_inches='tight', dpi=300)

    figure.clf()
    plt.close(figure)

def test_rna(id, mod, snr):
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))

    print("\nUsing RNA with id {}.".format(id))
    rna = join(str(rna_folder), "rna-" + id + ".h5")
    model = load_model(rna, compile=False)    
    print(model.summary())

    with open(join(data_folder, mod + '_features.pickle'), 'rb') as evaluating_data:
        data = pickle.load(evaluating_data)
        test_data_norm = normalize(data[snr][2].reshape(1,-1), norm='l2')

    print("input data:" + str(test_data_norm))
    start = time.time()
    predict = model.predict(test_data_norm)
    result = np.argmax(predict)
    print(modulations[result] + " with " + str(format(np.amax(predict)*100,'.5f')) + "% of confidence.")
    end = time.time()
    print(str(end - start))

def get_partial_layers_values(id, mod, snr, layer='all'):
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))

    print("\nUsing RNA with id {}.".format(id))
    rna = join(str(rna_folder), "rna-" + id + ".h5")
    model = load_model(rna, compile=False)    
    print(model.summary())

    with open(join(data_folder, mod + '_features.pickle'), 'rb') as evaluating_data:
        data = pickle.load(evaluating_data)
        test_data_norm = normalize(data[snr][0].reshape(1,-1), norm='l2')
    
    inp = model.input
    layers = [layer.output for layer in model.layers]
    functions = [backend.function([inp], [out]) for out in layers]

    layer_outputs = [func(test_data_norm) for func in functions]
    print(layer_outputs)
    with open('layers_output_' + str(mod) + '.txt', 'w') as f:
        f.write('input data:')
        f.write(str(test_data_norm) + "\n\n")
        for line in layer_outputs:
            f.write(str(line) + "\n")

if __name__ == '__main__':
    #evaluate_rna(id="1d48d172")
    #quantize_rna(id="100bc8b4")
    #get_partial_layers_values("1d48d172", 'QAM16', 10)
    test_rna("1d48d172", 'QAM16', 10)
    