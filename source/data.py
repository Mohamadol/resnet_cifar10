import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image


def c10(
    batch_size=128,
    buffer_size=50000,
    num_classes=10,
    validation_split=0.15,
    augment=True,
    valid_split=True,
):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = tf.one_hot(y_train.squeeze(), num_classes)
    y_test = tf.one_hot(y_test.squeeze(), num_classes)

    # Split training data into train and validation
    total_train_samples = x_train.shape[0]
    val_size = int(total_train_samples * validation_split) if valid_split == True else 1

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
    train_dataset = train_dataset.shuffle(buffer_size)
    if augment == True:
        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset pipeline
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Test dataset pipeline
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def save_image_int(image, filename, scaling_factor=12):
    """
    Saves an image tensor (or batch tensor) to a file in integer format.

    :param image: NumPy array of the image or batch tensor.
    :param filename: File name to save the image.
    :param scaling_factor: Factor to scale the image values.
    """
    with open(filename, "w") as file:
        for x in np.nditer(image, order="C"):
            file.write(str(int(x * (1 << scaling_factor))) + " ")
        file.write("\n")


def save_tensors(tensor, filename_prefix, single_images=False):
    """
    Saves tensors to files in integer format.

    :param tensor: Tensor to save.
    :param filename_prefix: Prefix for the file names.
    :param single_images: Whether to save each image as a separate file.
    """
    if single_images:
        for idx in range(tensor.shape[0]):
            save_image_int(tensor[idx], f"{filename_prefix}{idx + 1}_scale12.inp")
    else:
        save_image_int(tensor, f"{filename_prefix}_scale12.inp")


def main():
    _, _, testData = c10(batch_size=16)

    for images_batch, _ in testData.take(1):  # Take one batch
        images_tensor = images_batch.numpy()

        # Split tensors
        imagesBatch8_1 = images_tensor[:8]
        imagesBatch8_2 = images_tensor[8:]
        imagesBatch16 = images_tensor

        print("Saving tensors to files...")
        save_tensors(imagesBatch8_1, "resnet50_inpBatch8_1")
        save_tensors(imagesBatch8_2, "resnet50_inpBatch8_2")
        save_tensors(imagesBatch16, "resnet50_inpBatch16")
        save_tensors(images_tensor, "resnet50_inp", single_images=True)

        print("All tensors saved successfully!")


if __name__ == "__main__":
    main()
