import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from nets.layers.SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


class Net(nn.Module):
    def __init__(self, l2_norm_dump=False):
        super(Net, self).__init__()

        self.l2_norm_dump = l2_norm_dump

        self.conv1 = nn.Conv2d(3,96,(7, 7),(2, 2))
        self.relu1 = nn.ReLU()
        self.lrn1 = SpatialCrossMapLRN_temp(*(5, 0.0005, 0.75, 2))
        self.pool1 = nn.MaxPool2d((3, 3), (3, 3), (0, 0), ceil_mode=True)

        self.conv2 = nn.Conv2d(96, 256, (5, 5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.conv3 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((3, 3), (3, 3), (0, 0), ceil_mode=True)

        self.fc6 = nn.Linear(4608, 2048)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = Variable(self.lrn1.forward(x.data))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = x.view(x.size(0), -1)
        x = self.relu6(x)
        x = self.drop6(x)
        x = self.fc7(x)

        l2_norm = x.div(torch.norm(x, p=2, dim=1).repeat(x.size(1), 1).t())

        if self.l2_norm_dump:
            loss = l2_norm
            return loss, l2_norm

        anchors = l2_norm[0].expand([525,128])
        positives = tile(l2_norm[1:6],0,105)
        negatives = l2_norm[6:].expand(5,105,128).flatten().view(525,128)

        loss = F.triplet_margin_loss(anchors, positives, negatives, p=2, margin=0.2)

        return loss, l2_norm