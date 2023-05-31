import os, argparse, yaml
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from src.nets import build_model
from src.feeder import get_feeder
import time, tqdm

def main(config):
    net = build_model(config['model'])
    train_dataset, train_loader = get_feeder(config['dataset'], train=True)
    val_dataset, val_loader = get_feeder(config['dataset'], train=False)
    use_gpu = torch.cuda.is_available()
    pretrained_path = config['model']['pretrained']
    if pretrained_path is not None and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path)
        net.load_state_dict(ckpt)
        print("last checkpoint restored")
    else:
        print(f"[ERROR] Can't found pretrained_path: {pretrained_path}")
    if use_gpu:
        print("[INFO] training on GPU")
        net = net.cuda()
    optimizer = Adam(net.parameters(), lr=config['model']['lr'])
    criterion = CrossEntropyLoss()
    model_name = config['model']["model_name"]
    log_writer = SummaryWriter("logs", comment=model_name)
    checkpoints_path = "checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)
    for e in range(1, config['model']['epochs'] + 1):
        start_time = time.time()
        net.train()
        train_loss = []
        for x, y in tqdm.tqdm(train_loader):
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        net.eval()
        val_loss = []
        for x, y in tqdm.tqdm(val_loader):
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            val_loss.append(loss.item())
        log_writer.add_scalar(f'loss/train', np.mean(train_loss), e)
        log_writer.add_scalar(f'loss/val', np.mean(val_loss), e)
        time_elaps = time.time() - start_time
        print("[LOG] epoch: {}, train_loss={:.5f}, val_loss={:.5f}, training time: {:.5f}".format(e, np.mean(train_loss), np.mean(val_loss), time_elaps))
        torch.save(net.state_dict(), f"{checkpoints_path}/{model_name}_best_model.pth")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    config = yaml.safe_load(open(parser.parse_args().config))
    config = Munch(config)
    main(config)