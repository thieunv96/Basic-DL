import os, argparse, yaml
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from src.nets import build_model
from src.feeder import get_feeder
from sklearn.metrics import classification_report

def main(config):
    net = build_model(config['model'])
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
        net = net.cuda()
    ground_truth = []
    predictions = []
    for x, y in val_loader:
        if use_gpu:
            x = x.cuda()
        y_pred = net(x)
        y_pred = torch.argmax(y_pred, dim=1)
        ground_truth += list(y.numpy())
        if use_gpu:
            predictions += list(y_pred.cpu().detach().numpy())
        else:
            predictions += list(y_pred.detach().numpy())
    print(classification_report(ground_truth, predictions))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    config = yaml.safe_load(open(parser.parse_args().config))
    config = Munch(config)
    main(config)