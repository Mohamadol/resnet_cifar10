import matplotlib.pyplot as plt
import numpy as np


class LRScheduleTriangle:
    def __init__(
        self,
        initial_lr=0.008,
        peak_lr=0.12,
        final_lr=0.00008,
        warmup_epochs=40,
        total_epochs=60,
    ):
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def compute_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # Phase 1: Linear increase to peak LR, then decrease back to initial LR
            progress = epoch / (self.warmup_epochs / 2)
            if progress <= 1.0:  # First half: increasing
                lr = self.initial_lr + (self.peak_lr - self.initial_lr) * progress
            else:  # Second half: decreasing
                lr = self.peak_lr - (self.peak_lr - self.initial_lr) * (progress - 1.0)
        else:
            # Phase 2: Gradually reduce to final_lr
            remaining_epochs = self.total_epochs - self.warmup_epochs
            decay_progress = (epoch - self.warmup_epochs) / remaining_epochs
            lr = self.initial_lr - (self.initial_lr - self.final_lr) * decay_progress
        return lr

    def get_lr_list(self):
        """Generates a list of learning rates for all epochs."""
        return [self.compute_lr(epoch) for epoch in range(self.total_epochs)]


class LRScheduleCosineAnnealing:
    def __init__(
        self,
        initial_lr=0.008,
        peak_lr=0.12,
        final_lr=0.00008,
        warmup_epochs=40,
        total_epochs=60,
    ):
        """
        Implements Cosine Annealing with Warmup.

        Parameters:
            - initial_lr: Starting learning rate before warmup.
            - peak_lr: Maximum learning rate after warmup.
            - final_lr: Minimum learning rate at the end of cosine decay.
            - warmup_epochs: Number of epochs for the warmup phase.
            - total_epochs: Total number of training epochs.
        """
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def compute_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # Phase 1: Linear warmup to peak learning rate
            lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (
                epoch / self.warmup_epochs
            )
        else:
            # Phase 2: Cosine Annealing from peak_lr to final_lr
            cosine_decay_epochs = self.total_epochs - self.warmup_epochs
            decay_progress = (epoch - self.warmup_epochs) / cosine_decay_epochs
            lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (
                1 + np.cos(np.pi * decay_progress)
            )
        return lr

    def get_lr_list(self):
        """Generates a list of learning rates for all epochs."""
        return [self.compute_lr(epoch) for epoch in range(self.total_epochs)]


def plot_learning_rate_schedule(lr_list, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(lr_list) + 1), lr_list, marker="o", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Learning rate plot saved as '{save_path}'.")


# Main Function
if __name__ == "__main__":
    # Triangle Schedule
    triangle_schedule = LRScheduleTriangle(
        initial_lr=0.008,
        peak_lr=0.12,
        final_lr=0.0008,
        warmup_epochs=40,
        total_epochs=60,
    )
    triangle_lr_list = triangle_schedule.get_lr_list()
    print("Triangle Learning Rates:", triangle_lr_list)

    # Plot Triangle Schedule
    plot_learning_rate_schedule(
        triangle_lr_list,
        title="Learning Rate Schedule (Triangle)",
        save_path="figures/lr_schedule_triangle.png",
    )

    # Cosine Annealing Schedule
    cosine_schedule = LRScheduleCosineAnnealing(
        initial_lr=0.001,
        peak_lr=0.12,
        final_lr=0.000001,
        warmup_epochs=15,
        total_epochs=100,
    )
    cosine_lr_list = cosine_schedule.get_lr_list()
    print("Cosine Annealing Learning Rates:", cosine_lr_list)

    # Plot Cosine Annealing Schedule
    plot_learning_rate_schedule(
        cosine_lr_list,
        title="Cosine Annealing Learning Rate Schedule with Warmup",
        save_path="figures/lr_schedule_cosine.png",
    )
