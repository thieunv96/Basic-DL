from .lenet import Letnet

def build_model(model_cfg):
    if model_cfg['model_name'] == "Lenet":
        return Letnet()
    else:
        raise "Could not find your model"