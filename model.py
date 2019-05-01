import torchvision
import torch
import torch.nn as nn

def get_model(pretrained=False):
    model = torchvision.models.alexnet(pretrained)

    return model