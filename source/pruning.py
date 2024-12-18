import tensorflow as tf
import numpy as np
import bisect
import os


def find_index_of_smallest_larger(sorted_list, x):
    """
    Finds the index of the smallest value in a sorted list that is larger than x.

    Args:
        sorted_list (list): A list of numbers sorted in ascending order.
        x (float): The target value.

    Returns:
        int: The index of the smallest value larger than x, or None if no such value exists.
    """
    index = bisect.bisect_right(sorted_list, x)
    if index < len(sorted_list):
        return index
    return None  # No value larger than


class MagnitudePruning(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        dense_accuracy,
        pruning_target,
        patience,
        path,
        maxLr=1e-3,
        minLr=1e-7,
        initSparsity=0.0,
    ):
        self.model = model
        self.dense_accuracy = dense_accuracy
        self.current_accuracy = dense_accuracy
        self.accuracy_tolerance = 0.01
        self.final_pruning_target = pruning_target
        self.current_sparsity = initSparsity

        self.pruning_targets = self._gen_pruning_targets(pruning_target)
        self.target_index = 0
        for i, target in enumerate(self.pruning_targets):
            print(f"target {i+1}: {target}")

        self.patience = patience
        self.epoch_counter = 0
        self.maxLr = maxLr
        self.minLr = minLr
        self.lr_reduce_factor = 0.5
        self.lr_scheduler = None

        self.masks = [
            tf.Variable(tf.ones_like(weight), trainable=False)
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
            for weight in layer.trainable_weights
        ]

        self.path = path

        if initSparsity != 0.0:
            checkpoint_name = f"{self.path}/pruned_{int(initSparsity * 100)}.h5"

            if os.path.exists(checkpoint_name):
                self.model.load_weights(checkpoint_name)
                print(f"Loaded weights from checkpoint: {checkpoint_name}")
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_name}")

            # Find the index of the smallest pruning target larger than initSparsity
            self.target_index = find_index_of_smallest_larger(
                self.pruning_targets, initSparsity
            )

            # Ensure the index is valid
            if self.target_index is None or self.target_index >= len(
                self.pruning_targets
            ):
                raise ValueError(
                    f"Invalid target_index: {self.target_index}. Ensure initSparsity ({initSparsity}) is within the range of pruning_targets."
                )
            print(f"Starting from pruning target index: {self.target_index}")

    def _gen_pruning_targets(
        self, final_pruning_target, initial_target=0.25, num_points=10, power=2
    ):
        normalized_points = np.linspace(0, 1, num_points) ** power
        targets = (
            initial_target + (final_pruning_target - initial_target) * normalized_points
        )
        return [round(target, 4) for target in targets]

    def _reset_learning_rate(self):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.maxLr)

    def _update_masks(self, target_ratio):
        """
        Update masks only for Conv2D layer weights based on the global pruning threshold.
        """
        assert (
            0.0 <= target_ratio <= 1.0
        ), "target_ratio must be a fraction between 0 and 1"

        # Gather all Conv2D layer weights into a single tensor
        conv_weights = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                for weight in layer.trainable_weights:
                    conv_weights.append(tf.reshape(weight, [-1]))  # Flatten weights

        if not conv_weights:
            raise ValueError("No Conv2D layers found in the model.")

        all_weights = tf.concat(conv_weights, axis=0)  # Combine all Conv2D weights

        # Compute the global threshold
        sorted_weights = tf.sort(tf.abs(all_weights))
        k = tf.cast(tf.size(sorted_weights) * target_ratio, tf.int32) - 1
        global_threshold = sorted_weights[k] if k >= 0 else 0

        if global_threshold == 0:
            print(
                "Warning: Global threshold is zero, pruning might remove all weights!"
            )

        # Update masks for Conv2D weights
        conv_weight_index = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                for weight in layer.trainable_weights:
                    new_mask = tf.cast(tf.abs(weight) >= global_threshold, weight.dtype)
                    self.masks[conv_weight_index].assign(
                        new_mask
                    )  # Update mask in place
                    conv_weight_index += 1

    def _prune_weights(self):
        """
        Apply masks only to Conv2D layer weights to prune them.
        """
        conv_weight_index = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                for weight in layer.trainable_weights:
                    pruned_weights = weight * self.masks[conv_weight_index]
                    weight.assign(pruned_weights)
                    conv_weight_index += 1

    def on_batch_end(self, batch, logs=None):
        target_ratio = self.pruning_targets[self.target_index]
        if batch == 0 or (batch + 1) % 10 == 0:
            self._update_masks(target_ratio)
            self._prune_weights()
        else:
            self._prune_weights()

    def on_epoch_end(self, epoch, logs=None):
        self._prune_weights()

        total_params = 0
        zero_params = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                for weight in layer.trainable_weights:
                    total_params += tf.size(weight).numpy()
                    zero_params += tf.reduce_sum(tf.cast(weight == 0, tf.int32)).numpy()

        if total_params > 0:
            self.current_sparsity = zero_params / total_params
        else:
            self.current_sparsity = 0.0
        print(f"Conv2D Layers Sparsity: {self.current_sparsity:.4%}")

        self.current_accuracy = logs.get("val_accuracy")

        print(
            f"\nEpoch {epoch + 1}: Validation Accuracy = {self.current_accuracy:.4f} -- Sparsity = {self.current_sparsity}"
        )

        if (self.dense_accuracy - self.current_accuracy) <= self.accuracy_tolerance:
            print(
                f"Accuracy is within {self.accuracy_tolerance} of dense model. Moving to next pruning target."
            )

            # save the model
            current_pruning_ratio = self.pruning_targets[self.target_index]
            checkpoint_name = (
                f"{self.path}/pruned_{int(current_pruning_ratio * 100)}.h5"
            )
            self.model.save_weights(checkpoint_name)
            print(f"Model checkpoint saved as {checkpoint_name}")

            # update counters
            self.target_index += 1
            self.epoch_counter = 0

            # reset lr
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.maxLr)
            print(f"Learning rate reset to {self.maxLr:.7f}")

            # check if pruning is finished
            if self.target_index >= len(self.pruning_targets):
                print("All pruning targets achieved. Stopping training.")
                self.model.stop_training = True
            else:
                print(
                    f"New target index: {self.target_index}. Target ratio: {self.pruning_targets[self.target_index]}"
                )

        else:
            self.epoch_counter += 1
            if self.epoch_counter >= self.patience:
                current_lr = tf.keras.backend.get_value(
                    self.model.optimizer.learning_rate
                )
                new_lr = max(current_lr * self.lr_reduce_factor, self.minLr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(
                    f"Accuracy not recovered after {self.patience} epochs. Reducing learning rate to {new_lr:.7f}"
                )
            else:
                print(
                    f"Accuracy not recovered. Waiting for {self.patience - self.epoch_counter} more epochs before reducing LR."
                )

    def on_epoch_begin(self, epoch, logs=None):

        next_target_sparsity = (
            self.pruning_targets[self.target_index]
            if self.target_index < len(self.pruning_targets)
            else None
        )

        print(f"Epoch {epoch + 1} begins:")
        print(
            f" - Current Accuracy: {self.current_accuracy:.4f}, Original Accuracy: {self.dense_accuracy:.4f}"
            if self.current_accuracy
            else " - Current Accuracy: Not available"
        )
        print(f" - Current Sparsity: {self.current_sparsity:.4%}")
        print(
            f" - Next Target Sparsity: {next_target_sparsity:.4%}"
            if next_target_sparsity is not None
            else " - No further pruning targets."
        )
