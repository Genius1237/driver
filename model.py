import torchvision
import torch
import torch.nn as nn

model = torchvision.models.alexnet(pretrained=False)

t = torch.randn(1,3,360,640)
print(t.shape)
t = model.features(t)
print(t.shape)
t = model.avgpool(t)
print(t.shape)
t = t.view(-1,256*6*6)
t = model.classifier(t)
print(t.shape)
