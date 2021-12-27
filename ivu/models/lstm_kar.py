import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def lstm_kar_model(hidden_units=64, input_features=136, sequence_count=32, n_classes=7):
    _x_input = tf.keras.Input(shape=(None, input_features))
    _x = tf.transpose(_x_input, [1, 0, 2])
    # _x = tf.transpose(_x_input, [0, 2, 1, 3])

    _x = tf.reshape(_x, [-1, input_features])
    _x = tf.nn.relu(_x)
    _x = tf.split(_x, sequence_count, 0)

    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/BasicLSTMCell
    lstm_cell_1 = tf.keras.layers.LSTMCell(
        hidden_units, kernel_regularizer=tf.keras.regularizers.l2(l2=0.0015)
    )
    lstm_cell_2 = tf.keras.layers.LSTMCell(
        hidden_units, kernel_regularizer=tf.keras.regularizers.l2(l2=0.0015)
    )
    lstm_cells = tf.keras.layers.StackedRNNCells([lstm_cell_1, lstm_cell_2])

    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/static_rnn
    outputs, _ = tf.compat.v1.nn.static_rnn(lstm_cells, _x, dtype=tf.float32)

    lstm_last_output = outputs[-1]
    output = tf.keras.layers.Dense(n_classes, activation="linear")(lstm_last_output)
    # output = tf.keras.activations.linear(lstm_last_output)
    return tf.keras.Model(inputs=_x_input, outputs=output)
