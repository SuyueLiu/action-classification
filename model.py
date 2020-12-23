import torch
import torch.nn as nn


class ActNet(nn.Module):
    # VGG-16
    def __init__(self):
        super(ActNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 9 * 9, 1024),
            nn.Dropout2d(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Dropout2d(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        scores = self.fc2(x)

        return scores

    # def __init__(self):
    #     super(ActNet, self).__init__()
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(3, 32, 3, 1, 1),
    #         nn.ReLU()
    #     )
    #     self.fc1 = nn.Sequential(
    #         nn.Linear(300 * 300 * 32, 64),
    #         nn.Dropout(p=0.3),
    #         nn.ReLU(),
    #     )
    #     self.fc2 = nn.Sequential(
    #         nn.Linear(64, 128),
    #         nn.ReLU(),
    #     )
    #     self.fc3 = nn.Sequential(
    #         nn.Linear(128, 3),
    #     )
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     scores = self.fc3(x)
    #
    #     return scores