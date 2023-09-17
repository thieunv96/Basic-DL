from typing import Any, Optional
from torch import nn 
import lightning.pytorch as pl
from torchmetrics import Accuracy
import torch


class TinyVGG(pl.LightningModule):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
    def training_step(self, batch, batch_idx):
        loss, y_preds, y = self.common_step(batch, batch_idx)
        acc = self.accuracy(y_preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        # self.log_dict(
        #     {
        #         "train_loss":loss,
        #         "train_acc":acc
        #     },
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True
        # )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_preds, y = self.common_step(batch, batch_idx)
        acc = self.accuracy(y_preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_preds, y = self.common_step(batch, batch_idx)
        acc = self.accuracy(y_preds, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        loss, y_preds, y = self.common_step(batch, batch_idx)
        idx_preds = torch.argmax(y_preds, dim=1)
        return idx_preds

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.forward(x)
        loss = self.loss_fn(y_preds, y)
        return loss, y_preds, y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),  self.lr)
    