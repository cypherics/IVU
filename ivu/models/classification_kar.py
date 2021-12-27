import tensorflow as tf
from tensorflow.keras import layers


def classification_kar_model(n_classes=7):
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes, activation="relu"),
            # layers.Softmax()
        ]
    )
    return model
