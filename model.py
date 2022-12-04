import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self,in_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.drop_out = nn.Dropout()

    def forward(self, x):

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        return self.drop_out(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.pool(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, layers_num):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.layers_num = layers_num
        self.layers = self._layer_builder()


    def _layer_builder(self):

        in_put = self.in_channels
        layer_list = nn.ModuleList()
        for layer in range(self.layers_num):
            layer_list += [ConvBlock(in_channels=in_put)]
            in_put+=self.in_channels
        return  layer_list
    def forward(self, x):
         input = x
        for layer in self.layers:
            y = layer(x)
            x = torch.cat([x, y], dim=1)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        pass


def test():
