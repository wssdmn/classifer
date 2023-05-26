import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from myutils.weight_init import weightsInit


class MSMDGANetCnn(nn.Module):
    def __init__(self):
        super(MSMDGANetCnn, self).__init__()
        print("[ModelName: MSMDGANetCnn]")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(2*2*512, 800)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(800, 500)
        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)

        self.relu = nn.ReLU()
        self.leaklyrelu = nn.LeakyReLU()

    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        x = self.maxpooling1(x)
        # print(x.shape)  # [2, 64, 32, 32]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        x = self.maxpooling2(x)
        # print(x.shape)  # [2, 128, 16, 16]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        x = self.maxpooling3(x)
        # print(x.shape)  # [2, 256, 8, 8]

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        x = self.maxpooling4(x)
        # print(x.shape)  # [2, 512, 2, 2]

        x = x.reshape(x.size(0), -1)
        # print(x.shape)  # [2, 2048]
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        # print(x.shape)  # [2, 500]
        x = self.softmax(x)
        return x

class MSMDGANetCnn_wo_MaxPool(nn.Module):
    def __init__(self, out_channel):

        super(MSMDGANetCnn_wo_MaxPool, self).__init__()
        print("[ModelName: MSMDGANetCnn_wo_MaxPool]")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(5*5*512, 800)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(800, out_channel)
        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)

        self.relu = nn.ReLU()
        self.leaklyrelu = nn.LeakyReLU()

    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        # print(x.shape)  # [2, 64, 32, 32]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        # print(x.shape)  # [2, 128, 16, 16]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        # print(x.shape)  # [2, 256, 8, 8]

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # x = self.leaklyrelu(x)
        # print(x.shape)  # [2, 512, 5, 5]

        x = x.reshape(x.size(0), -1)
        # print(x.shape)  # [2, 2048]
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        # print(x.shape)  # [2, 500]
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from torchsummary import summary
    # classifier = MSMDGANetCnn().to("cuda")
    # summary(classifier, (1, 64, 64))

    classifier = MSMDGANetCnn_wo_MaxPool().to("cuda")
    summary(classifier, (1, 64, 64))

