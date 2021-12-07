import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import utils
import preprocessing

# DEVICE = '/GPU:0'
DEVICE = '/device:CPU:0'


def build_model(numClasses):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(numClasses, activation='relu'),
        # layers.Softmax()
    ])
    return model


def main():
    fpath = 'datasets/ds1_mediapipe.pkl'
    # fpath = 'datasets/ds2_actionai.pkl'
    dataset = utils.load_pickle(fpath)
    X, Y = dataset['X'], dataset['Y']
    X = X.reshape(X.shape[0], -1)
    X_train, Y_train, _, _, X_test, Y_test = preprocessing.split_dataset(X, Y)
    numClasses = len(dataset['labels'])
    del X, Y

    early_stop_fn = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = build_model(numClasses)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    result = model.fit(
        X_train, Y_train,
        epochs=200, batch_size=64, validation_split=0.1,
        callbacks=[early_stop_fn]
    )

    print('training loss  :', result.history['loss'][-1])
    print('validation loss:', result.history['val_loss'][-1])

    print('evaluating...')
    model.evaluate(X_test, Y_test)

    # Dataset1:
    # mediapipe: ~95%
    # ActionAi:  ~40%
    
    # Dataset2:
    # mediapipe: ~95%
    # ActionAi:  ~55%
    
    
if __name__ == '__main__':
    with tf.device(DEVICE):
        main()
