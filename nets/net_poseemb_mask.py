import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from nets.layers.masklayer import MaskLayer


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


class Net(nn.Module):
    def __init__(self, posebit_count):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(posebit_count, 4608)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4608, 2048)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(2048, 128)

        self.masklayer = MaskLayer()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        l2_norm = x.div(torch.norm(x, p=2, dim=1).repeat(x.size(1), 1).t())
        warped_l2_norm = self.masklayer(l2_norm)

        anchors = Variable(torch.zeros(128)).cuda().expand([3080,128])
        positives = tile(warped_l2_norm[0:55], 0, 56)
        negatives = warped_l2_norm[55:].expand(55,56,128).flatten().view(56*55,128)

        assert x.size(0) == 111

        loss = F.triplet_margin_loss(anchors, positives, negatives, p=2, margin=0.2)

        return loss, warped_l2_norm


if __name__ == '__main__':
    Net(200)