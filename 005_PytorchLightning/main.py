from src.feeder import MnistData
from src.nets import TinyVGG
from src.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method
import argparse

set_start_method("fork")
NUM_CLASS = 10
BATCH_SIZE = 512
EPOCHS = 15

def train():
    checkpoint_saver = ModelCheckpoint(dirpath="checkpoints", filename="best_checkpoint")
    trainer = pl.Trainer(min_epochs=1, max_epochs=EPOCHS, callbacks=[checkpoint_saver])
    model = TinyVGG(NUM_CLASS)
    dm = MnistData("./data", BATCH_SIZE, 0)
    trainer.fit(model, dm)
    trainer.validate(model, dm)

def test():
    trainer = pl.Trainer()
    model = TinyVGG(num_classes=NUM_CLASS)
    dm = MnistData("./data", BATCH_SIZE, 0)
    # load checkpoint
    checkpoint = torch.load("./checkpoints/best_checkpoint.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    trainer.test(model, dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='', help='Mode of functional, It should be is train or test.', required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()