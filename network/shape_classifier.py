import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ShapeClassifier(nn.Module):
    def __init__(self, sub_function, net):
        super().__init__()
        self.sub_function = sub_function
        self.online_encoder = net

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

    def forward(self, x):
        x = x.cpu().numpy()
        sub = torch.Tensor(self.sub_function(x, 1024)).to(device)
        with torch.no_grad():
            for p in self.online_encoder.parameters():
                p.requires_grad = False
            x = self.online_encoder(sub)
        x = x.reshape(x.shape[0], -1)  # bs, 1024
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
