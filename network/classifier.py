import torch
import torch.nn as nn
from torchvision import models


class Classifier(nn.Module):
    out_channel = 2048

    def __init__(self, module_type, class_num, pretrained=False, pretrained_path=None, map_location=None):
        super(Classifier, self).__init__()
        self.module_type = module_type
        if not pretrained:
            if module_type == "resnet-50":
                self.backbone = models.resnet50()
            elif module_type == "resnet-101":
                self.backbone = models.resnet101()
            elif module_type == "resnet-152":
                self.backbone = models.resnet152()
        else:
            self.backbone = torch.load(pretrained_path, map_location)
            self.backbone = self.backbone._modules["module"]

        del self.backbone.fc
        self.class_num = class_num
        self.fc_0 = nn.Linear(2048, 2048)
        self.bn_0 = nn.BatchNorm1d(2048)
        self.active = nn.ReLU()
        self.fc_1 = nn.Linear(2048, class_num)

    def forward(self, x: torch.Tensor, feature_weight=None):
        """
        :param x: (B, C, H, W)
        :param feature_weight: (B, C_0, H_0, W_0)
        """
        x = self.backbone.conv1(x)  # 1/2
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 1/4

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)  # 1/8
        x = self.backbone.layer3(x)  # 1/16
        x = self.backbone.layer4(x)  # 1/32

        if feature_weight is not None:
            x = x * feature_weight

        x = self.backbone.avgpool(x)
        extend = x.shape[0] == 1
        x = torch.squeeze(x)
        if extend:
            x = torch.unsqueeze(x, dim=0)
        feat = self.active(self.bn_0(self.fc_0(x)))
        x = self.fc_1(feat)

        return feat, x
