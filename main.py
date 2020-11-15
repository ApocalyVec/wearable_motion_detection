import math
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy import interpolate
from tensorflow.python.keras import models

from utils import window_slice, build_train_rnn, plot_train_history, plot_cm_results, build_train_cnn, build_train_ann, \
    build_train_birnn_with_attention, get_img_from_fig, plot_roc_multiclass
import matplotlib.pyplot as plt


# meta information

def up_sample(data, target_length):
    """

    :param data: of shape (legnth, NChannels)
    :param target_length:
    """
    assert len(data.shape) == 2
    datanew = np.zeros((target_length, data.shape[1]))
    for i, y in enumerate(np.transpose(data)):
        x = np.linspace(0, len(y), num=len(y))
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0, len(y) - 1, target_length)
        ynew = f(xnew)
        datanew[:, i] = np.transpose(ynew)
        # plt.plot(x, data[:, i], '-', xnew, datanew[:, i], '-')
        # plt.show()
    return datanew


def load_data(data_root, feature_names, scenario_data, data_all, labels_all):
    num_tags = 3
    label_index = 1
    scenario_indices = [True, False, True]
    # data information
    window_size = 200  # take 10 seconds as a sample
    stride = 5  # step every second because the sampling rate is 20 Hz
    downsample_threshold = 10000  # complete data has 160001 points and they need to be downsampled
    downsample_steps = 10

    features = [
        [os.path.join(data_root, ftn, fn) for fn in os.listdir(os.path.join(data_root, ftn)) if not fn.startswith('.')]
        for ftn in features_names]
    features = np.array(features).transpose()  # make all the features to be a row entry

    data_whole_seqs = []
    # load in the data
    for i, fls in enumerate(features):
        print('Processing file ' + str(i) + ' of ' + str(len(features)))
        dfs = [pd.read_excel(fl, header=None) for fl in fls]
        tags = [np.array(fl.strip('.xls').split('_')[-num_tags:]) for fl in fls]
        scenario = tuple(tags[0][scenario_indices])

        data = np.array([np.squeeze(df.values, axis=-1) for df in dfs]).transpose()
        if len(data) > downsample_threshold:  # downsample the large data
            data = data[::downsample_steps]
            print('data is of len ' + str(len(data)) + ' after downsampling')
        elif len(data) < window_size and len(
                data) >= window_size / 2:  # upsample by interpolate for too short sequences
            data = up_sample(data, target_length=window_size)
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
    return scenario_data, data_all, labels_all
    # build and train the models


features_names = ['pha', 'RSS']
scenario_data = {}
data_all = np.empty(shape=(0, len(features_names)))
labels_all = np.empty(shape=(0, 1))
data_roots = [
    '/media/apocalyvec/Extreme SSD/data/wmd/1012/Calibrated data for the experimenal data at 201903',
    '/media/apocalyvec/Extreme SSD/data/wmd/1012/Calibrated data for the experimenal data at 201906',
    '/media/apocalyvec/Extreme SSD/data/wmd']
for dr in data_roots:
    scenario_data, data_all, labels_all = load_data(dr, features_names, scenario_data, data_all, labels_all)

sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(data_all)
encoder = OneHotEncoder()
encoder.fit(labels_all)

scenario_train_histories = {}

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

    # test the BIRNN_attention
    history = build_train_birnn_with_attention(x_train, x_test, y_train, y_test)
    plot_train_history(history, note=str(scn) + ' BIRNN_attention')
    eval = history.model.evaluate(x=x_test, y=y_test)
    scenario_train_histories[('BIRNN_attention', scn)] = [history, eval]


strcts = {'ANN': build_train_ann, 'RNN': build_train_rnn, 'CNN': build_train_cnn, 'BIRNN_attention': build_train_birnn_with_attention}
freqs = ['04g', '24g', '5g']
locs = ['hea', 'han', 'foo']
# reorder
scenario_train_histories_reorder = {}
for L in locs:
    for F in freqs:
        for mdl in strcts.keys():
            scenario_train_histories_reorder[(mdl, (L, F))] = scenario_train_histories[(mdl, (L, F))]

# horizontal axis are the frequencies, vertical are the locations
for mdl in strcts.keys():
    plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories_reorder.items() if mdl in key], note=mdl)

# scenario_train_histories_without_model = list([mdl_scn, his_eval[0].history] for mdl_scn, his_eval in scenario_train_histories.items())
# pickle.dump(scenario_train_histories_without_model, open('results/scenario_train_histories_111420.p', 'wb'))










rename_map = {'foo': 'foot', 'han': 'hand', 'hea': 'head'}
rename_freq_map = {'04g': '0.4 GHz', '24g': '2.4 GHz', '5g': '5 GHz'}

freqs = ['0.4 GHz', '2.4 GHz', '5 GHz']


scenario_train_histories_rename = dict()
for key, value in scenario_train_histories.items():
    key_rename = (key[0], (rename_map[key[1][0]], rename_freq_map[key[1][1]]))
    scenario_train_histories_rename[key_rename] = value

# plot scenario-based performance
width = 0.175
plt.rcParams["figure.figsize"] = (8, 4)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
# for scns in [locs, freqs]:
#     strct_sns_acc_dict = dict([(model_strct, []) for model_strct in
#                                strcts.keys()])  # i.e., ('ANN', [0.9, 0.8, 0.3]) in the list are the average accuracy for this sencario, sencario could be 04g
#     for sn in scns:  # i.e., sn = 04g, scns = '04g', '24g', '5g', find marginal accuracy for all 04g
#         for model_strct in strcts.keys():  # iterate through model structures: ANN, RNN, CNN
#             sn_strct_accuracies = [hist_eval[1][1] for strct_scn, hist_eval in scenario_train_histories_rename.items()
#                                    if strct_scn[0] == model_strct and sn in strct_scn[1]]
#             sn_strct_acc = np.mean(sn_strct_accuracies)
#             strct_sns_acc_dict[model_strct].append(sn_strct_acc)
#     ind = np.arange(len(scns))
#     for i, model_strct_scn_accs in enumerate(strct_sns_acc_dict.items()):
#         model_strct, scn_accs = model_strct_scn_accs
#         plt.bar(ind + i * width, scn_accs, width, label=model_strct)
#         for j in range(len(scn_accs)):
#             plt.text(ind[j] + i * width, scn_accs[j] + 0.05, str(round(scn_accs[j], 3)), horizontalalignment='center',
#                      verticalalignment='center')
#     plt.ylim(0.0, 1.1)
#     plt.ylabel('Average Accuracy')
#     plt.title('Accuracy by model structure and scenario')
#     plt.xticks(ind + width / 2, scns)
#     plt.legend(loc=4)
#     plt.show()
#
# plot scenarios-based training history ##################################################################
plt.rcParams["figure.figsize"] = (15, 15)
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

fig, ax = plt.subplots(nrows=len(locs), ncols=len(freqs))
for i, lc in enumerate(locs):
    for j, fq in enumerate(freqs):
        strct_sn_hists = [(strct_scn[0], hist_eval[0].history['val_accuracy']) for strct_scn, hist_eval in
                          scenario_train_histories_rename.items() if (lc, fq) == strct_scn[1]]
        for strct, sn_hist in strct_sn_hists:
            ax[i][j].plot(sn_hist, label=strct)
        ax[i][j].set_xlabel('Epoch')
        ax[i][j].set_ylabel('Validation accuracy')
        # ax[i][j].set_ylim(-0.1, 3.0)
        ax[i][j].set_title(str(lc) + ', ' + str(fq))
        # ax[i][j].legend(loc='best')
plt.show()

# visualize intermidiate CNN kernels ############################
# first sample ###############
scn = ('han', '24g')
clss = 'sta'
np.random.seed(3)
dataset = scenario_data[scn]
sample = dataset['x'][np.random.choice(np.where(dataset['y'] == clss)[0])]
sample_normalized = sc.transform(sample)
sample_normalized_batch = np.expand_dims(sample_normalized, axis=0)

plt.rcParams["figure.figsize"] = (8, 4)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(sample[:, 0], color=color)
ax1.set_xlabel('Timesteps')
ax1.set_ylabel('Phase', color=color)


color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(sample[:, 1], color=color)
ax2.set_xlabel('Timesteps')
ax2.set_ylabel('RSS', color=color)

fig.tight_layout()
plt.title('Scenario: ' + rename_map[scn[0]] + ',' + rename_freq_map[scn[1]] + '; ' + clss)
plt.show()

# get corresponding model
model = scenario_train_histories[('CNN', scn)][0].model

# xy = scenario_data[scn]
# x = np.array([sc.transform(x_raw) for x_raw in xy['x']])
# y = encoder.transform(xy['y'].reshape(-1, 1)).toarray()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, shuffle=True)
# eval = model.evaluate(x=x_test, y=y_test)

classes = model.predict(sample_normalized_batch)
print("Predicted class is:", encoder.inverse_transform(classes))

# Creates a model that will return these outputs, given the model input
layer_outputs = [layer.output for layer in model.layers[:9] if 'batch_normalization' not in layer.name]
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(sample_normalized_batch)

layer_names = []
for layer in model.layers[:9]:
    if 'batch_normalization' not in layer.name:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.plot(first_layer_activation[0, :, 0])
plt.show()

images_per_row = 8
dpi = 180
sub_fig_size = 5
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = dpi * sub_fig_size

    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size, 3), dtype=int)
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, col * images_per_row + row]
            fig = plt.figure(figsize=(sub_fig_size, sub_fig_size))
            ax = fig.add_subplot(111)
            plt.plot(channel_image, linewidth = 3)
            plt.axis('off')
            channel_image = get_img_from_fig(fig, dpi=dpi)
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size, :] = channel_image
            plt.clf()
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
# visualize Attention weights  ############################

# plot ROC curve for biRNN model  ############################

plt.rcParams["figure.figsize"] = (15, 15)
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

fig, ax = plt.subplots(nrows=len(locs), ncols=len(freqs))
for i, scn_xy in enumerate(scenario_data.items()):
    scn, xy = scn_xy
    print('Plotting scenario: ' + str(scn))
    x = np.array([sc.transform(x_raw) for x_raw in xy['x']])
    y = encoder.transform(xy['y'].reshape(-1, 1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, shuffle=True)

    model = scenario_train_histories[('BIRNN_attention', scn)][0].model
    eval = model.evaluate(x=x_test, y=y_test)
    y_score = model.predict(x_test)

    plot_roc_multiclass(n_classes=3, y_score=y_score, y_test=y_test, ax=ax[((math.floor(i / 3)) % 3)][i % 3], zoom=True)
    # for strct, sn_hist in strct_sn_hists:
    #     ax[i][j].plot(sn_hist, label=strct)
    ax[((math.floor(i / 3)) % 3)][i % 3].set_xlabel('Epoch')
    ax[((math.floor(i / 3)) % 3)][i % 3].set_ylabel('Validation accuracy')
    ax[((math.floor(i / 3)) % 3)][i % 3].set_title(rename_map[scn[0]] + ', ' + rename_freq_map[scn[1]])
plt.show()