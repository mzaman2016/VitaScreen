import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class DiabetesRiskClassifier(L.LightningModule):
    def __init__(self, model, batch_size: int, pos_weight: float = None):
        super(DiabetesRiskClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = model
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).view(-1)  # Ensure 1D tensor for metrics and loss
        y = y.view(
            -1
        ).float()  # Ensure 1D tensor for metrics and loss and cast to float
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_f1",
            self.f1(y_hat, y.float()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).view(-1)
        y = y.view(
            -1
        ).float()  # Ensure 1D tensor for metrics and loss and cast to float
        loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_f1",
            self.f1(y_hat, y.float()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x).view(-1)
        y = y.view(
            -1
        ).float()  # Ensure 1D tensor for metrics and loss and cast to float
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_f1", self.f1(y_hat, y.float()), prog_bar=True)

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=8e-4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # because we want to minimize val_loss
            factor=0.5,  # reduce LR by half
            patience=2,  # wait 2 epochs with no improvement
            # verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # name must match what you log
                "interval": "epoch",
                "frequency": 1,
            },
        }
