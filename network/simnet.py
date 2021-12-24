import torch
import torch.nn as nn
from torchvision import models


class SimNet(nn.Module):
    def __init__(self, mode, backbone, pretrained, pretrained_path):
        super(SimNet, self).__init__()
        assert mode in ["train", "infer"]
        self.backbone = BackBone(backbone, pretrained, pretrained_path)
        self.enum = Enumerator()
        self.cross_ref = CRBlock(BackBone.out_channel)
        self.fusion = FusionBlock(2 * BackBone.out_channel)
        self.classifier = Classifier(2 * BackBone.out_channel)
        self.mode = mode
        self.n_feature = BackBone.out_channel

    def forward(self, x):
        x = self.backbone(x)
        x = self.enum(x)
        cr_feature = self.cross_ref(x)
        cls_feature, cls_res = self.classifier(self.fusion(cr_feature))
        if self.mode == "train":
            return cls_res, cls_feature
        elif self.mode == "infer":
            return cls_res, cr_feature

    def parallel(self, device_ids):
        self.cuda(device_ids[0])
        self.backbone = nn.DataParallel(self.backbone, device_ids=device_ids)
        self.cross_ref = nn.DataParallel(self.cross_ref, device_ids=device_ids)
        self.fusion = nn.DataParallel(self.fusion, device_ids=device_ids)
        self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids)
        return self

    def de_parallel(self):
        self.backbone = self.backbone.module
        self.cross_ref = self.cross_ref.module
        self.fusion = self.fusion.module
        self.classifier = self.classifier.module
        return self


class BackBone(nn.Module):
    out_channel = 2048

    def __init__(self, module_type, pretrained=False, pretrained_path=None):
        super(BackBone, self).__init__()
        self.module_type = module_type
        if not pretrained:
            if module_type == "resnet-50":
                self.backbone = models.resnet50()
            elif module_type == "resnet-101":
                self.backbone = models.resnet101()
            elif module_type == "resnet-152":
                self.backbone = models.resnet152()
        else:
            self.backbone = torch.load(pretrained_path)
            self.backbone = self.backbone._modules["module"]

        del self.backbone.fc
        del self.backbone.avgpool

    def forward(self, x: torch.Tensor):
        """
        :param x: has shape (batch_size, C, H, W)
        :return: the feature map with shape (batch_size, out_channel, H/32, W/32)
        """
        x = self.backbone.conv1(x)  # 1/2
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 1/4

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)  # 1/8
        x = self.backbone.layer3(x)  # 1/16
        x = self.backbone.layer4(x)  # 1/32

        return x


class Enumerator:
    def __call__(self, x: torch.Tensor):
        """
        Input [a, b, c], output [(a, a), (a, b), (a, c), (b, a), (b, b), ...]
        :param x: has shape (batch_size, C, H, W)
        :return: Tensor with shape (batch_size * batch_size, 2, C, H, W)
        """
        assert x.ndim == 4
        batch_size, c, h, w = x.shape
        x0 = torch.reshape(x.repeat(1, batch_size, 1, 1), (-1, c, h, w))
        x1 = x.repeat(batch_size, 1, 1, 1)
        return torch.stack([x0, x1], dim=1)


class SingleEnumerator:
    def __call__(self, x: torch.Tensor, batch_y: torch.Tensor):
        """
        concat x with batch_y
        :param x: (C, H, W) or (N0, C, H, W)
        :param batch_y: (N1, C, H, W)
        :return: Tensor with shape (N0 * N1, 2, C, H, W)
        """
        assert batch_y.ndim == 4
        batch_size, channel, height, width = batch_y.shape
        if x.ndim == 3:
            x = x[None, ...]
        batch_y = batch_y.repeat(x.shape[0], 1, 1, 1)
        x = x.repeat(1, batch_size, 1, 1).view(-1, channel, height, width)
        return torch.stack([x, batch_y], dim=1)


class CRBlock(nn.Module):
    """
    The cross-reference block
    """
    def __init__(self, channel):
        super(CRBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel = channel
        self.fc_0 = nn.Linear(self.channel, self.channel)
        self.fc_1 = nn.Linear(self.channel, self.channel)
        self.active = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor with shape (N, 2, C, H, W)
        :return: Tensor which has the same shape as input
        """
        assert x.ndim == 5
        extend = True if x.shape[0] == 1 else False

        x0_org, x1_org = x[:, 0, ...], x[:, 1, ...]
        x0 = torch.squeeze(self.avgpool(x0_org))
        x1 = torch.squeeze(self.avgpool(x1_org))
        if extend:
            x0 = torch.unsqueeze(x0, dim=0)
            x1 = torch.unsqueeze(x1, dim=0)
        # now x0 and x1 has the same shape (N, C)
        x0 = self.active(self.fc_0(x0))
        x1 = self.active(self.fc_1(x1))
        fuse = torch.unsqueeze(torch.unsqueeze(x0 * x1, dim=-1), -1)
        x0 = x0_org * fuse
        x1 = x1_org * fuse
        return torch.stack([x0, x1], dim=1)


class FusionBlock(nn.Module):
    def __init__(self, channel):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, (3, 3), (1, 1), 1)
        self.active = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        fuse the output from CRBlock
        :param x: Tensor with shape (N, 2, C, H, W)
        :return: Tensor with shape (N, 2 * C)
        """
        n, _, c, h, w = x.shape
        x = torch.reshape(x, (n, -1, h, w))
        x = self.conv(x)
        x = self.active(self.bn(x))
        x = self.avgpool(x)
        extend = True if x.shape[0] == 1 else False
        x = torch.squeeze(x)
        if extend:
            x = torch.unsqueeze(x, 0)
        return x


class Classifier(nn.Module):
    """
    The linear classifier, without Softmax at end
    """

    def __init__(self, n_feature):
        super(Classifier, self).__init__()
        self.fc_0 = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.BatchNorm1d(n_feature),
            nn.ReLU(inplace=True)
        )

        self.n_feature = int(n_feature / 2)
        self.fc_1 = nn.Sequential(
            nn.Linear(n_feature, self.n_feature),
            nn.BatchNorm1d(self.n_feature),
            nn.ReLU(inplace=True)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(self.n_feature, self.n_feature),
            nn.BatchNorm1d(self.n_feature),
            nn.ReLU(inplace=True)
        )

        self.fc_3 = nn.Linear(self.n_feature, 2)

    def forward(self, x):
        cls_feature = self.fc_1(self.fc_0(x))
        cls_res = self.fc_3(self.fc_2(cls_feature))
        return cls_feature, cls_res
