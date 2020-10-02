import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import window_slice, build_train_rnn, plot_train_history, plot_cm_results, build_train_cnn, build_train_ann
import matplotlib.pyplot as plt
# meta information
data_root = 'D:\PycharmProjects\wearable_motion_detection\data'
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

sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(data_all)
encoder = OneHotEncoder()
encoder.fit(labels_all)

scenario_train_histories= {}

for scn, xy in scenario_data.items():
    print('Training on scenario: ' + str(scn))
    x = np.array([sc.transform(x_raw) for x_raw in xy['x']])
    y = encoder.transform(xy['y'].reshape(-1, 1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, shuffle=True)

    # test the ANN
    history = build_train_ann(x_train, x_test, y_train, y_test)
    plot_train_history(history, note=str(scn) + ' ANN')
    eval = history.model.evaluate(x=x_test, y=y_test)
    scenario_train_histories[('ANN', scn)] = [history, eval]

    # test the RNN
    history = build_train_rnn(x_train, x_test, y_train, y_test)
    plot_train_history(history, note=str(scn) + ' RNN')
    eval = history.model.evaluate(x=x_test, y=y_test)
    scenario_train_histories[('RNN', scn)] = [history, eval]

    # test the CNN
    history = build_train_cnn(x_train, x_test, y_train, y_test)
    plot_train_history(history, note=str(scn) + ' CNN')
    eval = history.model.evaluate(x=x_test, y=y_test)
    scenario_train_histories[('CNN', scn)] = [history, eval]


freqs = ['04g', '24g', '5g']
locs = ['foot', 'hand', 'head']
# horizontal axis are the frequencies, vertical are the locations
mdl = 'ANN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)
mdl = 'RNN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)
mdl = 'CNN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)
