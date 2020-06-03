import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import window_slice, build_train_rnn, plot_train_history
import matplotlib.pyplot as plt
# meta information
data_root = 'D:\PycharmProjects\wearable_motion_detection\data'
features_names = ['pha', 'RSS']
num_tags = 3
label_index = 1
scenario_indices = [True, False, True]
# data information
window_size = 200  # take 10 seconds as a sample
stride = 10  # step every second because the sampling rate is 20 Hz

features = [[os.path.join(data_root, ftn, fn) for fn in os.listdir(os.path.join(data_root, ftn))] for ftn in features_names]
features = np.array(features).transpose()  # make all the features to be a row entry
scenario_data = {}
data_all = np.empty(shape=(0, len(features_names)))
labels_all = np.empty(shape=(0, 1))
# load in the data
for i, fls in enumerate(features):
    print('Processing file ' + str(i) + ' of ' + str(len(features)))
    dfs = [pd.read_excel(fl, header=None) for fl in fls]
    tags = [np.array(fl.strip('.xls').split('_')[-num_tags:]) for fl in fls]
    scenario = tuple(tags[0][scenario_indices])

    data = np.array([np.squeeze(df.values, axis=-1)for df in dfs]).transpose()
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

scenario_train_histories = {}
for scn, xy in scenario_data.items():
    print('Training on scenario: ' + str(scn))
    x = np.array([sc.transform(x_raw) for x_raw in xy['x']])
    y = encoder.transform(xy['y'].reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, shuffle=True)
    history = build_train_rnn(x_train, x_test, y_train, y_test)
    plot_train_history(history)
    eval = history.model.evaluate(x=x_test, y=y_test)
    scenario_train_histories[scn] = [history, eval]

# scenario_train_evals = dict([(scn, entry[1]) for scn, entry in scenario_train_histories.items()])
# pickle.dump(scenario_train_evals, open('scenario_train_histories_052320.p', 'wb'))
freqs = ['04g', '24g', '5g']
locs = ['foot', 'hand', 'head']
# horizontal axis are the frequencies, vertical are the locatiosn
acc_matrix = np.array([entry[1][1] for scn, entry in scenario_train_histories.items()]).reshape((3,3))

fig, ax = plt.subplots()
im = ax.imshow(acc_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(freqs)))
ax.set_yticks(np.arange(len(locs)))
# ... and label them with the respective list entries
ax.set_xticklabels(freqs)
ax.set_yticklabels(locs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(locs)):
    for j in range(len(freqs)):
        text = ax.text(j, i, round(acc_matrix[i, j], 3),
                       ha="center", va="center", color="w")

ax.set_title("Three motion classification accuracies")
# fig.tight_layout()
plt.show()