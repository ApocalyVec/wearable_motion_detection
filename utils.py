import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
import matplotlib.pylab as plt


def window_slice(data, window_size, stride):
    assert window_size <= len(data)
    assert stride > 0
    rtn = []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)


def build_train_rnn(x_train, x_test, y_train, y_test, epochs=100, batch_size=256):
    keras.backend.clear_session()
    regressor = Sequential()
    regressor.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform'))
    regressor.add(Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    regressor.add(LSTM(units=64, return_sequences=True, kernel_initializer='random_uniform'))
    regressor.add(Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    regressor.add(LSTM(units=64, return_sequences=False, kernel_initializer='random_uniform'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=128, kernel_initializer='random_uniform'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform'))
    adam = optimizers.adam(lr=1e-4, decay=1e-7)
    regressor.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = regressor.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=epochs, batch_size=batch_size)
    return history


def plot_train_history(history, note=''):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy ' + note)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss ' + note)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()