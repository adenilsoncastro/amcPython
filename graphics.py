import json
import os
import pathlib
import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

# Open json file with parameters
with open("./info.json") as handle:
    info_json = json.load(handle)

# Config
frame_size = info_json['frameSize']
number_of_modulations = len(info_json['modulations']['names'])
number_of_snr = len(info_json['snr']['using'])
snr_list = info_json['snr']['using']
number_of_frames = info_json['numberOfFrames']
number_of_features = len(info_json['features']['using'])
modulation_names = info_json['modulations']['names']
feature_names = info_json['features']['names']
data_set = info_json['dataSetForTraining']
mod_labels = ["BPSK", "QPSK", "PSK8", "QAM16", "NOISE"]

all_features = []
for modulation in range(number_of_modulations):
    # Filename setup
    if data_set == "rayleigh":
        pkl_file_name = pathlib.Path(join(os.getcwd(),
                                          'gr-data',
                                          'pickle',
                                          modulation_names[modulation] + '_features.pickle'))
    if data_set == "matlab":
        pkl_file_name = pathlib.Path(join(os.getcwd(),
                                          'mat-data',
                                          'pickle',
                                          modulation_names[modulation] + '_features.pickle'))

    # Load the pickle file
    with open(pkl_file_name, 'rb') as handle:
        all_features.append(pickle.load(handle))
    print('Files loaded...')

# Calculate mean, min and max of features by SNR and by modulation
features = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
mean_features = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
std_features = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
for m in range(number_of_modulations):
    for snr in range(number_of_snr):
        for ft in range(number_of_features):
            features[m, snr, :, ft] = all_features[m][snr, :, ft]
            mean_features[m, snr, :, ft] = np.mean(features[m][snr, :, ft])
            std_features[m, snr, :, ft] = np.std(features[m][snr, :, ft])
print('Means and standard deviations calculated...')

# SNR axis setup
#var = np.linspace((snr_list[0] - 10) * 2, (snr_list[-1] - 10) * 2, number_of_snr)
var = [info_json['snr']['values'][i] for i in snr_list]
snr_array = np.ndarray([number_of_modulations, number_of_snr, number_of_frames, number_of_features])
for m in range(number_of_modulations):
    for snr in range(number_of_snr):
        for fr in range(number_of_frames):
            for ft in range(number_of_features):
                snr_array[m, snr, fr, ft] = var[snr]

# Plot graphics using only mean
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    plt.grid(b=True)
    # Plot without for loop because of bug
    # TODO: fix bug using for loop to plot (result is graphics with same color)
    plt.plot(snr_array[0, :, n, n], mean_features[0, :, n, n], '#03cffc', linewidth=1.0)  # BPSK
    plt.plot(snr_array[1, :, n, n], mean_features[1, :, n, n], '#6203fc', linewidth=1.0)  # QPSK
    plt.plot(snr_array[2, :, n, n], mean_features[2, :, n, n], '#be03fc', linewidth=1.0)  # PSK8
    plt.plot(snr_array[3, :, n, n], mean_features[3, :, n, n], '#fc0320', linewidth=1.0)  # QAM16
    #plt.plot(snr_array[4, :, n, n], mean_features[4, :, n, n], 'g', linewidth=1.0)  # QAM64
    plt.plot(snr_array[4, :, n, n], mean_features[4, :, n, n], 'k', linewidth=1.0)  # Noise
    plt.title('Feature ' + str(n + 1) + ' - ' +  feature_names[info_json['features']['using'][n]])
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.legend(mod_labels)
    if data_set == "rayleigh":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'feature_{}_SNR_({})_a_({})_means.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    if data_set == "matlab":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'matlab_feature_{}_SNR_({})_a_({})_means.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()
    print('Plotting means of feature number {}'.format(n))

# Plot graphics with all frames
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    plt.grid(b=True)
    plt.plot(snr_array[0, :, :, n], features[0][:, :, n], '#03cffc', linewidth=1.0)  # BPSK
    plt.plot(snr_array[1, :, :, n], features[1][:, :, n], '#6203fc', linewidth=1.0)  # QPSK
    plt.plot(snr_array[2, :, :, n], features[2][:, :, n], '#be03fc', linewidth=1.0)  # PSK8
    plt.plot(snr_array[3, :, :, n], features[3][:, :, n], '#fc0320', linewidth=1.0)  # QAM16
    #plt.plot(snr_array[4, :, :, n], features[4][:, :, n], 'g', linewidth=1.0)  # QAM64
    plt.plot(snr_array[4, :, :, n], features[4][:, :, n], 'k', linewidth=1.0)  # Noise
    plt.xlabel('SNR')
    plt.ylabel('Value')
    plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[info_json['features']['using'][n]])
    # TODO: put modulation names in legend
    #plt.legend(mod_labels)
    if data_set == "rayleigh":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'feature_{}_SNR_({})_a_({})_multiple_frames.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    if data_set == "matlab":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'matlab_feature_{}_SNR_({})_a_({})_multiple_frames.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()
    print('Plotting 500 frames of feature number {}'.format(n))

# Plot graphics with error bar using standard deviation
for n in range(number_of_features):
    plt.figure(num=n, figsize=(6.4, 3.6), dpi=300)
    plt.grid(b=True)
    plt.errorbar(snr_array[0, :, n, n],
                 mean_features[0, :, n, n],
                 yerr=std_features[0, :, n, n]*3, color='#03cffc')
    plt.errorbar(snr_array[1, :, n, n],
                 mean_features[1, :, n, n],
                 yerr=std_features[1, :, n, n]*3, color='#6203fc')
    plt.errorbar(snr_array[2, :, n, n],
                 mean_features[2, :, n, n],
                 yerr=std_features[2, :, n, n]*3, color='#be03fc')
    plt.errorbar(snr_array[3, :, n, n],
                 mean_features[3, :, n, n],
                 yerr=std_features[3, :, n, n]*4, color='#fc0320')
    #plt.errorbar(snr_array[4, :, n, n],
     #            mean_features[4, :, n, n],
     #            yerr=std_features[4, :, n, n], color='g')
    plt.errorbar(snr_array[4, :, n, n],
                 mean_features[4, :, n, n],
                 yerr=std_features[4, :, n, n]*3, color='k')
    plt.xlabel('SNR')
    plt.ylabel('Average Value')
    plt.title('Feature ' + str(n + 1) + ' - ' + feature_names[info_json['features']['using'][n]])
    plt.legend(mod_labels)
    if data_set == "rayleigh":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'feature_{}_SNR_({})_a_({})_means_with_stddev.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    if data_set == "matlab":
        figure_name = pathlib.Path(join(os.getcwd(), 'figures', 'features',
                                        'matlab_feature_{}_SNR_({})_a_({})_means_with_stddev.png'.format(str(n + 1),
                                                                                      snr_list[0],
                                                                                      snr_list[-1])))
    plt.savefig(figure_name, figsize=(6.4, 3.6), dpi=300)
    plt.close()
    print('Plotting error bar of feature number {}'.format(n))
