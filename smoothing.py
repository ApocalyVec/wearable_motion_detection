import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import window_slice, build_train_rnn, plot_train_history, plot_cm_results, build_train_cnn, build_train_ann
import matplotlib.pyplot as plt
# meta information
data_root = '/media/apocalyvec/Extreme SSD/data/wmd'
features_names = ['pha', 'RSS']
num_tags = 3
label_index = 1
scenario_indices = [True, False, True]
# data information
window_size = 200  # take 10 seconds as a sample
stride = 5  # step every second because the sampling rate is 20 Hz
downsample_threshold = 10000  # complete data has 160001 points and they need to be downsampled
downsample_steps = 10

features = [[os.path.join(data_root, ftn, fn) for fn in os.listdir(os.path.join(data_root, ftn))] for ftn in features_names]
features = np.array(features).transpose()  # make all the features to be a row entry
scenario_data = {}
data_all = np.empty(shape=(0, len(features_names)))
labels_all = np.empty(shape=(0, 1))
data_whole_seqs = []
# load in the data
for i, fls in enumerate(features):
    print('Processing file ' + str(i) + ' of ' + str(len(features)))
    dfs = [pd.read_excel(fl, header=None) for fl in fls]
    tags = [np.array(fl.strip('.xls').split('_')[-num_tags:]) for fl in fls]
    scenario = tuple(tags[0][scenario_indices])

    data = np.array([np.squeeze(df.values, axis=-1)for df in dfs]).transpose()
    if len(data) > downsample_threshold:  # downsample the large data
        data = data[::downsample_steps]
        print('data is of len ' + str(len(data)) + ' after downsampling')
    data_whole_seqs.append(data)

    samples = window_slice(data, window_size=window_size, stride=stride)
    label = [tags[0][label_index]] * len(samples)

    if scenario not in scenario_data.keys():
        scenario_data[scenario] = {'x': samples, 'y': label}
    else:
        scenario_data[scenario]['x'] = np.concatenate([scenario_data[scenario]['x'], samples])
        scenario_data[scenario]['y'] = np.concatenate([scenario_data[scenario]['y'], label])
    data_all = np.concatenate([data_all, data])
    labels_all = np.concatenate([labels_all, np.array([label]).transpose()])
# build and train the models




import numpy as np
from scipy import signal

# order = 5
# sampling_freq = 20
# cutoff_freq = 18
# sampling_duration = 10
# number_of_samples = sampling_freq * sampling_duration
# time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)
#
# normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
# numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)
# filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, d)

cutoff_f = 18
scn = ('han', '5g')

d = scenario_data[scn]['x']
d = d[800][:, 0]

t = np.linspace(0, 10, 200, False)

sos = signal.butter(10, cutoff_f, 'lowpass', fs=200, output='sos')
filtered = signal.sosfilt(sos, d)

plt.plot(d, label='Original')
plt.plot(filtered, label='Filtered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(str(scn))
plt.show()
