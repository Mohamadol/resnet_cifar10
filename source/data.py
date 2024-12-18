import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def c10(batch_size=128, buffer_size=50000, num_classes=10, validation_split=0.15):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = tf.one_hot(y_train.squeeze(), num_classes)
    y_test = tf.one_hot(y_test.squeeze(), num_classes)

    # Split training data into train and validation
    total_train_samples = x_train.shape[0]
    val_size = int(total_train_samples * validation_split)

    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    def augment(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 36, 36)  # Add padding
        image = tf.image.random_crop(image, [32, 32, 3])  # Random crop back to 32x32
        image = tf.image.random_flip_left_right(image)
        radians = np.deg2rad(15)  # random rotate up to 15-degree
        image = tfa.image.rotate(
            image, angles=tf.random.uniform(shape=[], minval=-radians, maxval=radians)
        )
        return image, label

    # Training dataset pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation dataset pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Test dataset pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
