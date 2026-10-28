"""WRN-28-10 CIFAR backbone for the L2RW uniform-flip experiment."""
import torch
from torch import nn
from torch.nn import functional as F
from .feature_output import FeatureOutput


class WideBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride, dropout):
        super().__init__()
        self.bn1=nn.BatchNorm2d(incoming)
        self.conv1=nn.Conv2d(incoming,outgoing,3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(outgoing)
        self.conv2=nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False)
        self.shortcut=nn.Conv2d(incoming,outgoing,1,stride=stride,bias=False) if incoming!=outgoing or stride!=1 else None
        self.dropout=float(dropout)

    def forward(self, x):
        activated=F.relu(self.bn1(x))
        shortcut=x if self.shortcut is None else self.shortcut(activated)
        value=self.conv1(activated)
        value=F.dropout(F.relu(self.bn2(value)),p=self.dropout,training=self.training)
        return self.conv2(value)+shortcut


class WideResNet28(nn.Module):
    def __init__(self,num_classes=100,width_multiplier=10,dropout=0.3):
        super().__init__()
        widths=[16,16*width_multiplier,32*width_multiplier,64*width_multiplier]
        self.stem=nn.Conv2d(3,16,3,padding=1,bias=False)
        layers=[]
        incoming=16
        for stage,outgoing in enumerate(widths[1:]):
            for block in range(4):
                layers.append(WideBlock(incoming,outgoing,2 if stage and block==0 else 1,dropout))
                incoming=outgoing
        self.blocks=nn.Sequential(*layers)
        self.bn=nn.BatchNorm2d(widths[-1])
        self.classifier=nn.Linear(widths[-1],num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):nn.init.zeros_(m.bias)

    def forward_with_features(self,x):
        value=F.relu(self.bn(self.blocks(self.stem(x))))
        features=F.adaptive_avg_pool2d(value,1).flatten(1)
        return FeatureOutput(self.classifier(features),features)

    def forward(self,x):return self.forward_with_features(x).logits
