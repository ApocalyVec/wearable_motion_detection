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

strcts = {'ANN':build_train_ann, 'RNN': build_train_rnn, 'CNN': build_train_cnn}
freqs = ['04g', '24g', '5g']
locs = ['foot', 'hand', 'head']
# horizontal axis are the frequencies, vertical are the locations
mdl = 'ANN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)
mdl = 'RNN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)
mdl = 'CNN'
plot_cm_results(freqs, locs, [(key, value) for key, value in scenario_train_histories.items() if mdl in key], note=mdl)

pickle.dump(scenario_train_histories, open('scenario_train_histories_10012020.p', 'wb'))

rename_map = {'foo': 'foot', 'han': 'hand', 'hea': 'head'}
scenario_train_histories_rename = dict()
for key, value in scenario_train_histories.items():
    key_rename = (key[0], (rename_map[key[1][0]], key[1][1]))
    scenario_train_histories_rename[key_rename] = value

# plot scenario-based performance
width = 0.175
plt.rcParams["figure.figsize"] = (8,4)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
for scns in [locs, freqs]:
    strct_sns_acc_dict = dict([(model_strct, []) for model_strct in
                               strcts.keys()])  # i.e., ('ANN', [0.9, 0.8, 0.3]) in the list are the average accuracy for this sencario, sencario could be 04g
    for sn in scns:  # i.e., sn = 04g, scns = '04g', '24g', '5g', find marginal accuracy for all 04g
        for model_strct in strcts.keys():  # iterate through model structures: ANN, RNN, CNN
            sn_strct_accuracies = [hist_eval[1][1] for strct_scn, hist_eval in scenario_train_histories_rename.items() if strct_scn[0] == model_strct and sn in strct_scn[1]]
            sn_strct_acc = np.mean(sn_strct_accuracies)
            strct_sns_acc_dict[model_strct].append(sn_strct_acc)
    ind = np.arange(len(scns))
    for i, model_strct_scn_accs in enumerate(strct_sns_acc_dict.items()):
        model_strct, scn_accs = model_strct_scn_accs
        plt.bar(ind + i * width, scn_accs, width, label=model_strct)
        for j in range(len(scn_accs)):
            plt.text(ind[j] + i * width, scn_accs[j] + 0.05, str(round(scn_accs[j], 3)), horizontalalignment='center', verticalalignment='center')
    plt.ylim(0.0, 1.1)
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy by model structure and scenario')
    plt.xticks(ind + width / 2, scns)
    plt.legend(loc=4)
    plt.show()



# plot scenarios-based training history
plt.rcParams["figure.figsize"] = (10, 10)

fig, ax = plt.subplots(nrows=len(locs), ncols=len(freqs))
for i, lc in enumerate(locs):
    for j, fq in enumerate(freqs):
        strct_sn_hists = [(strct_scn[0], hist_eval[0].history['val_loss']) for strct_scn, hist_eval in scenario_train_histories_rename.items() if (lc, fq) == strct_scn[1]]
        for strct, sn_hist in strct_sn_hists:
            ax[i][j].plot(sn_hist, label=strct)
        ax[i][j].set_xlabel('Epoch')
        ax[i][j].set_ylabel('Validation loss')
        ax[i][j].set_title(str(lc) + ', ' + str(fq))
        ax[i][j].legend(loc='best')
plt.show()
