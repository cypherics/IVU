import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def lstm_kar_model(
    hidden_units=64, input_features=136, n_classes=7, penalty=1e-04, **kwargs
):
    # [Batch, Number of frames, input features]
    _x_input = tf.keras.Input(shape=(None, input_features))

    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/BasicLSTMCell
    lstm_cell_1 = tf.keras.layers.LSTMCell(
        hidden_units, kernel_regularizer=tf.keras.regularizers.l2(l2=penalty)
    )
    lstm_cell_2 = tf.keras.layers.LSTMCell(
        hidden_units, kernel_regularizer=tf.keras.regularizers.l2(l2=penalty)
    )
    lstm_cells = tf.keras.layers.StackedRNNCells([lstm_cell_1, lstm_cell_2])

    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/static_rnn
    lstm_last_output = tf.keras.layers.RNN(lstm_cells)(_x_input)
    output = tf.keras.layers.Dense(n_classes, activation="linear")(lstm_last_output)
    # output = tf.keras.activations.linear(lstm_last_output)
    return tf.keras.Model(inputs=_x_input, outputs=output)
