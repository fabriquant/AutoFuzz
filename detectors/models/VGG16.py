import torch
import torch.nn as nn
from torchvision.models import vgg16


class VGGnet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16_features = list(vgg16(pretrained=False).features)
        features = vgg16_features[:31]
        features.append(Flatten())
        features.append(nn.Linear(in_features=5120,out_features=1024,bias=True))
        features.append(nn.Linear(in_features=1024,out_features=128,bias=True))
        self.feature = nn.Sequential(*list(features))

        self.classifier = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feature = self.feature(x)       # Sequential_1
        pred = self.classifier(feature)       # Dropout -> Dense -> Activation
        return pred, feature

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
