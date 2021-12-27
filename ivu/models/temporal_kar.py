import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def basic_residual_block(x_input, in_units, out_units, kernel=3, stride=2):
    h = tf.keras.layers.Conv1D(
        out_units, kernel_size=kernel, strides=stride, padding="SAME", use_bias=False
    )(x_input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)

    h = tf.keras.layers.Conv1D(
        out_units, kernel_size=kernel, strides=1, padding="SAME", use_bias=False
    )(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)

    h = tf.keras.layers.Conv1D(
        out_units, kernel_size=kernel, strides=1, padding="SAME", use_bias=False
    )(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)

    if stride != 1 or in_units != out_units:
        x_input = tf.keras.layers.Conv1D(
            out_units, kernel_size=1, strides=stride, padding="VALID", use_bias=False
        )(x_input)
        x_input = tf.keras.layers.BatchNormalization()(x_input)
    h = tf.keras.layers.Add()([h, x_input])
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    return h


def temporal_model(input_features=136, n_classes=7):
    # [Batch, Number of frames, input features]
    # input_shape = (3, 32, 581)
    # x = tf.random.normal(input_shape)

    # first value is dynamically set for number for frames
    x = tf.keras.Input(shape=(None, input_features))

    x_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding="SAME")(
        x
    )
    x_1 = tf.keras.layers.BatchNormalization()(x_1)
    x_1 = tf.keras.layers.Activation(tf.keras.activations.relu)(x_1)

    x_2 = basic_residual_block(x_1, in_units=256, out_units=256, kernel=3, stride=2)
    x_3 = basic_residual_block(x_2, 256, 256, kernel=3, stride=2)
    x_4 = basic_residual_block(x_3, 256, 256, kernel=3, stride=2)
    x_5 = basic_residual_block(x_4, 256, 256, kernel=3, stride=2)

    x_6 = tf.keras.layers.GlobalAveragePooling1D()(x_5)
    x_6 = tf.expand_dims(x_6, axis=1)
    x_7 = tf.keras.layers.Conv1D(filters=n_classes, kernel_size=1, strides=1)(x_6)
    x_7 = tf.squeeze(x_7, -2, name="concat")
    return tf.keras.Model(inputs=x, outputs=x_7)
