import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from myutils.weight_init import weightsInit

# Finger Vein Recognition Algorithm Based on Lightweight Deep Convolutional Neural Network
class LightweightDeepConvNN(nn.Module):
    def __init__(self, out_channel):
        super(LightweightDeepConvNN, self).__init__()
        print("[ModelName: LightweightDeepConvNN] [out_channel:{}]".format(out_channel))

        self.stemBlock = StemBlock()
        self.stageblock1 = StageBlock(in_c=32)
        self.stageblock2 = StageBlock(in_c=64)
        self.stageblock3 = StageBlock(in_c=96, output_layer=True)

        self.fc_out = nn.Linear(in_features=128*4*4, out_features=out_channel)

        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)


    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.stemBlock(x)   # [B, 32, 16, 16]

        x = self.stageblock1(x)
        x = self.stageblock2(x)
        x = self.stageblock3(x)
        # print(x.shape)   # [2, 128, 4, 4]

        x = x.reshape(x.size(0), -1)   # [2, 128 * 4 * 4]

        x = self.fc_out(x)

        x = self.softmax(x)
        return x

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.stem1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn_stem1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(16)

        self.stem3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.bn_stem3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.stem1(x)
        x = self.bn_stem1(x)
        x = self.relu(x)

        x1 = self.maxpool(x)

        x2 = self.conv1(x)
        x2 = self.bn_conv1(x2)
        x2 = self.relu(x2)

        x2 = self.conv2(x2)
        x2 = self.bn_conv2(x2)
        x2 = self.relu(x2)

        x = torch.cat((x1, x2), dim=1)

        x = self.stem3(x)
        x = self.bn_stem3(x)
        x = self.relu(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, in_c, output_layer=False):
        super(StageBlock, self).__init__()
        self.smallStage1 = SmallStageBlock(in_c=in_c)
        self.smallStage2 = SmallStageBlock(in_c=in_c + 8)
        self.smallStage3 = SmallStageBlock(in_c=in_c + 16)
        self.smallStage4 = SmallStageBlock(in_c=in_c + 24)   # 最后输出的通道数目是in_c+32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels=in_c+32, out_channels=in_c+32, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_c+32)
        self.isOutput_layer = output_layer

    def forward(self, x):
        x = self.smallStage1(x)
        x = self.smallStage2(x)
        x = self.smallStage3(x)
        x = self.smallStage4(x)
        if self.isOutput_layer:
            x = self.conv1x1(x)
            x = self.bn(x)
        else:
            x = self.pool(x)
        # print(x.shape)   # [B, in_c+32, H/2, W/2]
        return x


class SmallStageBlock(nn.Module):
    def __init__(self, in_c):
        super(SmallStageBlock, self).__init__()
        self.branch1_conv1 = nn.Conv2d(in_channels=in_c, out_channels=4, kernel_size=1)
        self.branch1_bn_conv1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()

        self.branch1_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.branch1_bn_conv2 = nn.BatchNorm2d(4)

        self.branch3_conv1 = nn.Conv2d(in_channels=in_c, out_channels=4, kernel_size=1)
        self.branch3_bn_conv1 = nn.BatchNorm2d(4)

        self.branch3_conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.branch3_bn_conv2 = nn.BatchNorm2d(4)

        self.branch3_conv3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.branch3_bn_conv3 = nn.BatchNorm2d(4)

    def forward(self, x):
        x1 = self.branch1_conv1(x)
        x1 = self.branch1_bn_conv1(x1)
        x1 = self.relu(x1)
        x1 = self.branch1_conv2(x1)
        x1 = self.branch1_bn_conv2(x1)
        x1 = self.relu(x1)
        # print(x1.shape)

        x3 = self.branch3_conv1(x)
        x3 = self.branch3_bn_conv1(x3)
        x3 = self.relu(x3)
        x3 = self.branch3_conv2(x3)
        x3 = self.branch3_bn_conv2(x3)
        x3 = self.relu(x3)
        x3 = self.branch3_conv3(x3)
        x3 = self.branch3_bn_conv3(x3)
        x3 = self.relu(x3)
        # print(x3.shape)

        x = torch.cat((x1, x, x3), dim=1)
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

    # maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    # a = torch.randn((2, 1, 16, 16))
    # b = maxpool(a)   # [2, 1, 8, 8]
    # print(b.shape)

    # a1 = torch.randn((2, 1, 16, 16))
    # a2 = torch.randn((2, 1, 16, 16))
    # b = torch.cat((a1, a2), dim=1)
    # print(b.shape)   # [2, 2, 16, 16]

    # stemBlock = StemBlock()
    # a = torch.randn((2, 1, 64, 64))
    # b = stemBlock(a)
    # print(b.shape)   # [2, 32, 16, 16]

    # smallStageBlock = SmallStageBlock(in_c=32)
    # a = torch.randn((2, 32, 16, 16))
    # b = smallStageBlock(a)
    # print(b.shape)  # [2, 40, 16, 16]

    # stageBlock = StageBlock(in_c=32)
    # a = torch.randn((2, 32, 16, 16))
    # b = stageBlock(a)
    # print(b.shape)  # [2, 64, 8, 8]

    classifier = LightweightDeepConvNN().to("cuda")
    summary(classifier, (1, 64, 64))


