from lightning.pytorch.callbacks import Callback


class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Extract metrics from the trainer's logger after the validation epoch ends
        train_loss = trainer.callback_metrics.get("train_loss_epoch", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        train_f1 = trainer.callback_metrics.get("train_f1", None)
        val_f1 = trainer.callback_metrics.get("val_f1", None)

        # Only append metrics if they exist
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        if train_f1 is not None:
            self.train_accs.append(train_f1.item())

        if val_f1 is not None:
            self.val_accs.append(val_f1.item())

        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        train_f1_str = f"{train_f1:.3f}" if train_f1 is not None else "N/A"
        val_f1_str = f"{val_f1:.3f}" if val_f1 is not None else "N/A"

        # Clean print
        print(
            f"Epoch {trainer.current_epoch}: "
            f"train_loss={train_loss_str} | val_loss={val_loss_str} | "
            f"train_f1={train_f1_str} | val_f1={val_f1_str}"
        )
