import numpy as np
import matplotlib.pylab as plt
from tensorflow.python.keras.backend import clear_session
import tensorflow as tf
from matplotlib.pyplot import cm
from tensorflow.python.keras.layers import CuDNNLSTM, Concatenate

from attention import Attention


def window_slice(data, window_size, stride):
    assert window_size <= len(data)
    assert stride > 0
    rtn = np.expand_dims(data, axis=0) if window_size == len(data) else []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)

def build_train_rnn(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    classifier.add(tf.keras.layers.LSTM(units=64, return_sequences=True, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    classifier.add(tf.keras.layers.LSTM(units=64, return_sequences=False, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(units=128, kernel_initializer='random_uniform'))
    classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform'))
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

def build_train_ann(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()

    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1:])))
    classifier.add(tf.keras.layers.Dense(units=128, kernel_initializer='random_uniform'))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))

    classifier.add((tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='random_uniform')))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))
    classifier.add((tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform')))

    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history


def build_train_cnn(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3, ), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Flatten())

    classifier.add((tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='random_uniform')))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))
    classifier.add((tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform')))

    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history



def build_train_birnn_with_attention(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()

    sequence_input = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(64, return_sequences=True), name="bi_lstm_0")(sequence_input)
    lstm = tf.keras.layers.Dropout(0.2, name="drop_0")(lstm)
    # lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(64, return_sequences=True), name="bi_lstm_1")(lstm)
    # lstm = tf.keras.layers.Dropout(0.2, name="drop_1")(lstm)
    # lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(64, return_sequences=True), name="bi_lstm_2")(lstm)
    # lstm = tf.keras.layers.Dropout(0.2, name="drop_2")(lstm)
    (lstm, forward_h, forward_c, backward_h, backward_c) = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, return_state=True), name="bi_lstm_1")(lstm)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    context_vector, attention_weights = Attention(10)(lstm, state_h)
    dense1 = tf.keras.layers.Dense(128, activation="relu")(context_vector)
    dropout = tf.keras.layers.Dropout(0.05)(dense1)
    output = tf.keras.layers.Dense(y_train.shape[1], activation="softmax")(dropout)

    classifier = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

    # classifier = tf.keras.Sequential()
    # classifier.add(tf.keras.layers.Bidirectional(CuDNNLSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))))
    # classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    # classifier.add(tf.keras.layers.Bidirectional(CuDNNLSTM(units=64, return_sequences=True, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))))
    # classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    # classifier.add(tf.keras.layers.Bidirectional(CuDNNLSTM(units=64, return_sequences=True, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))))
    # classifier.add(tf.keras.layers.Dropout(0.2))
    # classifier.add(tf.keras.layers.Attention())
    # classifier.add(tf.keras.layers.Dense(units=128, kernel_initializer='random_uniform'))
    # classifier.add(tf.keras.layers.Dropout(0.2))
    # classifier.add(tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform'))
    # adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    # classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

def plot_train_history(history, note=''):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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

def plot_cm_results(freqs, locs, train_histories, note=''):
    acc_matrix = np.array([entry[1][1] for scn, entry in train_histories]).reshape((3, 3))

    fig, ax = plt.subplots()
    im = ax.imshow(acc_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(freqs)))
    ax.set_yticks(np.arange(len(locs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(freqs, size=18)
    ax.set_yticklabels(locs, size=18)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(locs)):
        for j in range(len(freqs)):
            text = ax.text(j, i, round(acc_matrix[i, j], 3),
                           ha="center", va="center", color="black", size=20)

    ax.set_title("Three motion classification accuracies " + note, size=15)
    # fig.tight_layout()
    plt.show()