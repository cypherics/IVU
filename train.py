import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import utils
import preprocessing

# DEVICE = '/GPU:0'
DEVICE = '/device:CPU:0'


def buildModel(numClasses):
    # model = tf.keras.Sequential([
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(numClasses, activation='relu'),
    #     # layers.Softmax()
    # ])
    
    model = tf.keras.Sequential([
        layers.Dense(128, activation=tf.nn.relu6),
        layers.Dropout(0.5),
        layers.Dense(128, activation=tf.nn.relu6),
        layers.Dropout(0.5),
        layers.Dense(numClasses, activation='softmax'),
    ])

    return model


def main():
    dsfpaths = [os.path.join('datasets', x) for x in os.listdir('datasets') if not x.startswith('pose')]
    datasets = {key: utils.load_pickle(key) for key in dsfpaths}

    evalData = dict()

    for fpath in dsfpaths:
        print(fpath)

        dataset1 = datasets[fpath]
        otherFpath = fpath.replace('ds1', 'ds2') if 'ds1' in fpath else fpath.replace('ds2', 'ds1')
        dataset2 = datasets[otherFpath]

        data = {
            'train': list(),
            'valid': list(),
            'trainCM': list(),
            'test': list(),
            'testCM': list(),
            'testOther': list(),
            'testOtherCM': list(),
        }

        loops = 10
        for i in range(loops):
            print(' ', i, '/', loops, end='\r')

            numClasses = len(dataset1['labels'])
            X_train = dataset1['X_train'].reshape(dataset1['X_train'].shape[0], -1)
            Y_train = dataset1['Y_train'].reshape(dataset1['Y_train'].shape[0], -1)

            # build the model
            model = buildModel(numClasses)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # train the model
            early_stop_fn = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
            result = model.fit(
                X_train, Y_train, shuffle=True,
                epochs=500, batch_size=64, validation_split=0.1,
                callbacks=[early_stop_fn], verbose=False
            )
            data['train'].append(result.history['accuracy'][-1])
            data['valid'].append(result.history['val_accuracy'][-1])

            # confusion matrix of training data
            Y_pred = model(X_train).numpy()
            Y_pred = np.argmax(Y_pred, axis=1)
            cm = tf.math.confusion_matrix(Y_train, Y_pred).numpy()
            data['trainCM'].append(cm)

            # test current dataset
            X_test = dataset1['X_test'].reshape(dataset1['X_test'].shape[0], -1)
            Y_test = dataset1['Y_test'].reshape(dataset1['Y_test'].shape[0], -1)
            Y_pred = model(X_test).numpy()
            Y_pred = np.argmax(Y_pred, axis=1)
            cm1 = tf.math.confusion_matrix(Y_test, Y_pred).numpy()
            acc1 = (np.sum(np.diag(cm1)) / Y_pred.shape[0]) * 100
            data['test'].append(acc1)
            data['testCM'].append(cm1)
            
            # test other dataset (use the train data of the other dataset to have more samples)
            X_train = dataset2['X_train'].reshape(dataset2['X_train'].shape[0], -1)
            Y_train = dataset2['Y_train'].reshape(dataset2['Y_train'].shape[0], -1)
            Y_pred = model(X_train).numpy()
            Y_pred = np.argmax(Y_pred, axis=1)
            cm2 = tf.math.confusion_matrix(Y_train, Y_pred).numpy()
            acc2 = (np.sum(np.diag(cm2)) / Y_pred.shape[0]) * 100
            data['testOther'].append(acc2)
            data['testOtherCM'].append(cm2)

        evalData[fpath] = data

    utils.save_pickle(evalData, 'evalData.pkl')
    print('done')

    
if __name__ == '__main__':
    with tf.device(DEVICE):
        main()
