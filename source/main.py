import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from source.data import c10
from source.res18 import resnet18
from source.utils import LRScheduleTriangle, LRScheduleCosineAnnealing
from source.pruning import MagnitudePruning

flags = {
    "train": False,
    "test": True,
    "prune": True,
}

trainInfo = {
    "epochs": 100,
    "epochsWarmup": 20,
    "batchSize": 128,
    "lrInit": 0.004,
    "lrPeak": 0.12,
    "lrEnd": 0.000001,
    "path": "models/dense.h5",
}


pruningInfo = {
    "targetRatio": 0.9,
    "initSparsity": 0.0,
    "epochs": 150,
    "epochsWarmup": 0,
    "batchSize": 128,
    "lrInit": 1e-3,
    "lrEnd": 1e-7,
    "lrPatience": 5,
    "path": "models",
}


def trainModel(
    model,
    train_dataset,
    test_dataset,
):

    # lr_schedule = LRScheduleTriangle(
    #     initial_lr=trainInfo["lrInit"],
    #     peak_lr=trainInfo["lrPeak"],
    #     final_lr=trainInfo["lrEnd"],
    #     warmup_epochs=trainInfo["epochsWarmup"],
    #     total_epochs=trainInfo["epochs"],
    # )
    lr_schedule = LRScheduleCosineAnnealing(
        initial_lr=trainInfo["lrInit"],
        peak_lr=trainInfo["lrPeak"],
        final_lr=trainInfo["lrEnd"],
        warmup_epochs=trainInfo["epochsWarmup"],
        total_epochs=trainInfo["epochs"],
    )
    learning_rates = lr_schedule.get_lr_list()

    def lr_scheduler(epoch, lr):
        return learning_rates[epoch]

    lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=trainInfo["path"],
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    optimizer = tf.keras.optimizers.SGD(momentum=0.9, weight_decay=5e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=trainInfo["epochs"],
        batch_size=trainInfo["batchSize"],
        callbacks=[
            lr_callback,
            checkpointCallback,
        ],
        verbose=1,
    )
    return


def testModel(
    model,
    test_dataset,
):
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, weight_decay=5e-4)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def pruneModel(model, trainData, validData, testData):

    lrWarmup = pruningInfo["lrEnd"]
    optimizer = tf.keras.optimizers.SGD(learning_rate=lrWarmup, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Warm-up phase
    print("Starting warm-up phase for 6 epochs with a very low learning rate.")
    warmup_epochs = pruningInfo["epochsWarmup"]
    model.fit(
        trainData,
        epochs=warmup_epochs,
        validation_data=validData,
        verbose=1,
    )

    # Pruning phase
    _, dense_accuracy = model.evaluate(validData, verbose=0)
    print(f"Dense model validation accuracy: {dense_accuracy:.4f}")

    tf.keras.backend.set_value(model.optimizer.learning_rate, pruningInfo["lrInit"])
    pruning_callback = MagnitudePruning(
        model=model,
        dense_accuracy=dense_accuracy,
        pruning_target=pruningInfo["targetRatio"],
        patience=pruningInfo["lrPatience"],
        path=pruningInfo["path"],
        maxLr=pruningInfo["lrInit"],
        minLr=pruningInfo["lrEnd"],
        initSparsity=pruningInfo["initSparsity"],
    )

    print("Starting pruning phase.")
    model.fit(
        trainData,
        epochs=150,
        validation_data=validData,
        callbacks=[pruning_callback],
        verbose=1,
    )
    print("Pruning completed.")


def main():
    train_dataset, val_dataset, test_dataset = c10()
    res18 = resnet18()

    if flags["train"]:
        trainModel(res18, train_dataset, val_dataset)
    else:
        res18.load_weights(trainInfo["path"])

    if flags["test"]:
        testModel(res18, test_dataset)

    if flags["prune"]:
        pruneModel(res18, train_dataset, val_dataset, test_dataset)


main()
