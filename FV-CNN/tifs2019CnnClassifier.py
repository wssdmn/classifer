import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

#from myutils.weight_init import weightsInit

# Convolutional Neural Network for Finger-Vein-Based Biometric Identificatio
class Tifs2019CnnClassifier(nn.Module):
    def __init__(self):
        super(Tifs2019CnnClassifier, self).__init__()
        print("[ModelName: Tifs2019Cnn]")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(512)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(768)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=500, kernel_size=1)

        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)


    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)
        # print(x.shape)  # [2, 153, 30, 30]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)
        # print(x.shape)  # [2, 512, 13, 13]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpooling3(x)
        # print(x.shape)  # [2, 768, 4, 4]

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu1(x)
        # print(x.shape)  # [2, 1024, 1, 1]

        x = self.conv5(x)
        # print(x.shape)  # [2, 500, 1, 1]
        x = x.squeeze(3).squeeze(2)
        x = self.softmax(x)
        return x


class Tifs2019CnnWithoutMaxPool(nn.Module):
    def __init__(self, out_channel):
        super(Tifs2019CnnWithoutMaxPool, self).__init__()
        print("[ModelName: Tifs2019Cnn_wo_MaxPool]")   # "Tifs2019Cnn_wo_MaxPool"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(768)

        self.conv4 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=out_channel, kernel_size=2)

        self.softmax = nn.Softmax(-1)
        self.apply(weightsInit)


    def forward(self, x):
        """
        :param x: b, 1, h, w
        :return: b, 1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.maxpooling1(x)
        # print(x.shape)  # [2, 153, 30, 30]

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.maxpooling2(x)
        # print(x.shape)  # [2, 512, 13, 13]

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.maxpooling3(x)
        # print(x.shape)  # [2, 768, 5, 5]

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu1(x)
        # print(x.shape)  # [2, 1024, 2, 2]

        x = self.conv5(x)
        # print(x.shape)  # [2, 500, 1, 1]
        x = x.squeeze(3).squeeze(2)
        x = self.softmax(x)
        return x


class ShallowTifs2019CnnClassifier(nn.Module):
    def __init__(self, model_name):
        super(ShallowTifs2019CnnClassifier, self).__init__()
        self.model_name = model_name
        print("[ModelName: {}]".format(model_name))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(512)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(768)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=768, out_channels=500, kernel_size=4)

        self.relu = nn.ReLU()

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
        x = self.maxpooling1(x)
        # print(x.shape)  # [2, 153, 30, 30]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpooling2(x)
        # print(x.shape)  # [2, 512, 13, 13]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpooling3(x)
        # print(x.shape)  # [2, 768, 5, 5]

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # print(x.shape)  # [2, 1024, 1, 1]

        x = self.conv5(x)
        # print(x.shape)  # [2, 500, 1, 1]
        x = x.squeeze(3).squeeze(2)
        x = self.softmax(x)
        return x


class ShallowTifs2019CnnClassifier_wo_maxpooling(nn.Module):
    def __init__(self, model_name):
        super(ShallowTifs2019CnnClassifier_wo_maxpooling, self).__init__()
        self.model_name = model_name
        print("[ModelName: {}]".format(model_name))
        # channels = [128, 512, 768]
        channels = [64, 128, 256]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])

        self.conv3 = nn.Conv2d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[1])

        self.dropout2 = nn.Dropout(0.5)

        # self.out = nn.Conv2d(in_channels=channels[2], out_channels=500, kernel_size=5)
        self.out = nn.Linear(channels[1]*8*8, 500)

        self.relu = nn.ReLU()

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
        # x = self.maxpooling1(x)
        # print(x.shape)  # [2, 153, 30, 30]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.maxpooling2(x)
        # print(x.shape)  # [2, 512, 13, 13]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.maxpooling3(x)
        # print(x.shape)  # [2, 768, 5, 5]

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # print(x.shape)  # [2, 1024, 1, 1]

        x = self.dropout2(x)

        x = x.reshape(x.size(0), -1)
        x = self.out(x)
        # print(x.shape)  # [2, 500, 1, 1]
        # x = x.squeeze(3).squeeze(2)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from torchsummary import summary
    # classifier = Tifs2019CnnClassifier(model_name="Tifs2019Cnn").to("cuda")
    # summary(classifier, (1, 64, 64))

    # classifier = Tifs2019CnnWithoutMaxPool(model_name="Tifs2019Cnn_wo_MaxPool").to("cuda")
    # summary(classifier, (1, 64, 64))

    classifier = ShallowTifs2019CnnClassifier_wo_maxpooling(model_name="ShallowTifs2019Cnn_wo_maxpooling").to("cuda")
    summary(classifier, (1, 64, 64))