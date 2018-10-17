import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, posebit_count):
        super(Net, self).__init__()

        # self.fc1 = nn.Linear(213, 1000)
        self.fc1 = nn.Linear(posebit_count, 4608)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        # self.fc2 = nn.Linear(1000, 2048)
        # self.fc2 = nn.Linear(18432, 2048)
        self.fc2 = nn.Linear(4608, 2048)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(2048, 128)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)

        # l2_norm = x.div(torch.norm(x, p=2, dim=1).repeat(1,x.size(1)))
        l2_norm = x.div(torch.norm(x, p=2, dim=1).repeat(x.size(1), 1).t())

        mm = torch.mm(l2_norm, y.t())
        diag = torch.diag(mm)

        loss = torch.mean(torch.abs(1 - diag))

        return loss, l2_norm