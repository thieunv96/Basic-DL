from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


def get_feeder(dataset_cfg, train=True):
    batch_size = dataset_cfg["train_batch_size"]
    trans = transforms.Compose([
        transforms.ToTensor()
    ]) 
    if dataset_cfg['dataset_name'] == "MNIST":
        ds = MNIST('data', train=train, transform=trans, download=True)
    elif dataset_cfg['dataset_name'] == "CIFAR10":
        ds = CIFAR10('data', train=train, transform=trans, download=True)
    else:
        raise "Could not find your dataset"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return ds, loader