import torch.nn as nn


class Discriminator(nn.Module):
    """
    The discriminator to judge whether a feature comes from the base dataset or
    from the novel dataset, no softmax at the end.
    """
    def __init__(self, n_feature):
        super(Discriminator, self).__init__()

        self.fc_0 = nn.Linear(n_feature, n_feature)
        self.bn = nn.LayerNorm(n_feature)
        self.active = nn.ReLU(inplace=True)
        self.fc_1 = nn.Linear(n_feature, 2)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.bn(x)
        x = self.active(x)
        x = self.fc_1(x)
        return x

    def parallel(self, gpu_ids):
        return nn.DataParallel(self, gpu_ids)
