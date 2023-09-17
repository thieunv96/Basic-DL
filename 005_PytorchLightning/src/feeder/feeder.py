import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn, optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split


class MnistData(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage) -> None:
        val_ratio = 0.2
        train_phase_data = datasets.MNIST(self.data_dir, train=True, transform=transforms.ToTensor(), download=False)
        num_val_data = int(val_ratio*len(train_phase_data))
        self.train_ds, self.val_ds = random_split(train_phase_data, [len(train_phase_data) - num_val_data, num_val_data])
        self.test_ds = datasets.MNIST(self.data_dir, train=False, transform=transforms.ToTensor(), download=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader (self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)