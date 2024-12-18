from tensorflow.keras import layers, models
import tensorflow as tf


def resnet18(input_shape=(32, 32, 3), num_classes=10):
    def conv_block(x, filters, kernel_size, strides, activation=True):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        if activation:
            x = layers.ReLU()(x)
        return x

    def residual_block(x, filters, strides=1):
        shortcut = x
        if strides != 1 or x.shape[-1] != filters:
            shortcut = conv_block(
                x, filters, kernel_size=1, strides=strides, activation=False
            )
        x = conv_block(x, filters, kernel_size=3, strides=strides)
        x = conv_block(x, filters, kernel_size=3, strides=1, activation=False)
        x = layers.Add()([x, shortcut])
        return layers.ReLU()(x)

    inputs = tf.keras.Input(shape=input_shape)
    x = conv_block(inputs, 64, kernel_size=3, strides=1)
    x = residual_block(x, 64, strides=1)
    x = residual_block(x, 64, strides=1)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128, strides=1)
    x = residual_block(x, 256, strides=1)
    x = residual_block(x, 256, strides=1)
    x = residual_block(x, 512, strides=2)
    x = residual_block(x, 512, strides=1)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)
