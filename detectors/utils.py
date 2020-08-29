import torch
from models.VGG16 import VGGnet

def save_model(net, path):
    torch.save(net.state_dict(), path)


def load_model(path, arch):
    if arch == 'vgg':
        net = VGGnet()

    net.load_state_dict(torch.load(path))
    net.cuda()

    return net
