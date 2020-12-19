import torch
import torch.nn as nn
import torchvision.models as models

class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(1, 6, 5, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(6, 16, 5, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.fc = nn.Linear(16*4*4, 10)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(1, 8, 5, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 12, 5, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
                    nn.Linear(12*4*4, 50),
                    nn.ReLU(),
                    nn.Linear(50, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyDeepModel(nn.Module):
    def __init__(self):
        super(MyDeepModel, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(1, 5, 3, 1),
                            nn.ReLU(),
                            nn.Conv2d(5, 8, 3, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 12, 3, 1),
                            nn.ReLU(),
                            nn.Conv2d(12, 16, 3, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
                    nn.Linear(16*4*4, 50),
                    nn.ReLU(),
                    nn.Linear(50, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GrayResnet18(models.resnet.ResNet):
    def __init__(self):
        super(GrayResnet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x