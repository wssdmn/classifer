import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from myutils.weight_init import weightsInit

# FVRAS-Net: An Embedded Finger-Vein Recognition and AntiSpoofing System Using a Unified CNN
class FVRASNet(nn.Module):
    def __init__(self):
        super(FVRASNet, self).__init__()
        print("[ModelName: FVRASNet]")
        channels = [64, 128, 256]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.block1 = ConvBlock(in_c=32, out_c=channels[0])
        self.block2 = ConvBlock(in_c=channels[0], out_c=channels[1])
        self.block3 = ConvBlock(in_c=channels[1], out_c=channels[2])

        self.fc1 = nn.Linear(in_features=256 * 2 * 2, out_features=256)

        self.fc_out = nn.Linear(in_features=256, out_features=500)

        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)


    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)   # [2, 32, 31, 31]

        x = self.block1(x)
        # print(x.shape)   # [2, 64, 14, 14]

        x = self.block2(x)
        # print(x.shape)   # [2, 128, 6, 6]

        x = self.block3(x)
        # print(x.shape)   # [2, 256, 2, 2]

        x = x.reshape(x.size(0), -1)   # [2, 256 * 2 * 2]

        x = self.fc1(x)
        x = self.fc_out(x)

        x = self.softmax(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_c, out_channels=out_c,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv1x1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpooling(x)
        return x

class FVRASNet_wo_Maxpooling(nn.Module):
    def __init__(self, out_channel):
        super(FVRASNet_wo_Maxpooling, self).__init__()
        print("[ModelName: FVRASNet_wo_Maxpooling]")
        channels = [64, 128, 256]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.block1 = ConvBlock_wo_Maxpooling(in_c=32, out_c=channels[0])
        self.block2 = ConvBlock_wo_Maxpooling(in_c=channels[0], out_c=channels[1])
        self.block3 = ConvBlock_wo_Maxpooling(in_c=channels[1], out_c=channels[2])

        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=256)

        self.fc_out = nn.Linear(in_features=256, out_features=out_channel)

        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)


    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)   # [2, 32, 32, 32]

        x = self.block1(x)
        # print(x.shape)   # [2, 64, 16, 16]

        x = self.block2(x)
        # print(x.shape)   # [2, 128, 8, 8]

        x = self.block3(x)
        # print(x.shape)   # [2, 256, 4, 4]

        x = x.reshape(x.size(0), -1)   # [2, 256 * 4 * 4]

        x = self.fc1(x)
        x = self.fc_out(x)

        x = self.softmax(x)
        return x

class ConvBlock_wo_Maxpooling(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock_wo_Maxpooling, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_c, out_channels=out_c,kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv1x1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from torchsummary import summary
    # classifier = FVRASNet().to("cuda")
    # summary(classifier, (1, 64, 64))

    # classifier = Tifs2019CnnWithoutMaxPool(model_name="Tifs2019Cnn_wo_MaxPool").to("cuda")
    # summary(classifier, (1, 64, 64))
    # m1 = nn.MaxPool2d(kernel_size=2)
    # a = torch.randn((1, 8, 8))
    # b = m1(a)
    # print(b.shape)

    # cb1 = ConvBlock_wo_Maxpooling(in_c=1, out_c=3)
    # a = torch.randn((2, 1, 64, 64))
    # b = cb1(a)
    # print(b.shape)

    classifier = FVRASNet_wo_Maxpooling().to("cuda")
    summary(classifier, (1, 64, 64))
