import os, argparse, yaml
from munch import Munch
import torch
import cv2
import numpy as np
from src.nets import build_model
from src.feeder import get_feeder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def stich(data_vis):
    N, H, W, C = data_vis.shape
    # find number of w
    n_w = 1
    while(N / (n_w + 1) > n_w * 2):
        n_w += 1
    n_h = N // n_w
    image = np.zeros((n_h * H, n_w * W, 3), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            idx = i * n_w + j
            image[i*H:i*H + H, j*W :j*W + W] = data_vis[idx]
    return image

def get_batch_original_image(x):
    ar_image = x.numpy()
    org_im = [cv2.cvtColor(np.transpose(im, (1,2,0)), cv2.COLOR_GRAY2RGB) for im in ar_image]
    org_im = np.array(org_im)
    return org_im

def main(config):
    use_gpu = torch.cuda.is_available()
    net = build_model(config['model'])
    target_layer = [net.bb]
    cam = GradCAM(model=net, target_layers=target_layer, use_cuda=use_gpu)
    _, val_loader = get_feeder(config['dataset'], train=False)
    pretrained_path = config['model']['pretrained']
    if pretrained_path is not None and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path)
        net.load_state_dict(ckpt)
        print("last checkpoint restored")
    else:
        print(f"[ERROR] Can't found pretrained_path: {pretrained_path}")
    if use_gpu:
        net = net.cuda()
    for x, y in val_loader:
        if use_gpu:
            x = x.cuda()
        y_pred = net(x)
        y_pred = torch.argmax(y_pred, dim=1)
        # get heatmap visualization
        targets = [ClassifierOutputTarget(i.item()) for i in y_pred]
        grayscale_cam = cam(input_tensor=x, targets=targets)
        rgb_imgs = get_batch_original_image(x)
        for rgb_im, gray_cam in zip(rgb_imgs, grayscale_cam):
            vis = show_cam_on_image(rgb_im, gray_cam, use_rgb=True)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            # cv2.imshow("vis", vis)
            # cv2.waitKey(0) 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    config = yaml.safe_load(open(parser.parse_args().config))
    config = Munch(config)
    main(config)