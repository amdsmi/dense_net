import torch.nn as nn
import torch
import config as cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        print(in_channels)
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.drop_out = nn.Dropout()

    def forward(self, x):
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
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
        # self.conv_block = ConvBlock(in_channels*layers_num)

    def _layer_builder(self):

        in_put = self.in_channels
        layer_list = nn.ModuleList()
        for layer in range(self.layers_num):
            layer_list += [ConvBlock(in_channels=in_put)]
            in_put += self.in_channels
        return layer_list

    def forward(self, x):
        residuals = [x]
        for layer in self.layers:
            y = layer(torch.cat(residuals, dim=1))
            residuals.append(y)
        return y


class DenseNet(nn.Module):
    def __init__(self, image_channels=3, class_num=10):
        super(DenseNet, self).__init__()
        self.image_channels = image_channels

        self.structure = cfg.layers

        self.body, out_channels = self._layer_builder()

        self.init_conv = nn.Conv2d(image_channels,
                                   image_channels,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3)
        self.bn = nn.BatchNorm2d(image_channels)

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(out_channels, class_num)

    def _layer_builder(self):

        layer_list = nn.ModuleList()
        in_channels = self.image_channels

        for layer in self.structure:
            if layer[0] == 'D':
                layer_list += [DenseBlock(in_channels, layer[1])]
                in_channels = in_channels * layer[1]
            elif layer[0] == 'T':
                layer_list += [TransitionBlock(in_channels)]

        return layer_list, in_channels

    def forward(self, x):

        x = self.relu(self.bn(self.init_conv(x)))

        x = self.max_pool(x)

        for layer in self.body:
            x = layer(x)

        x = self.avg_pool(x)

        return self.fc(x)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).to(device)
    net = DenseNet().to(device)
    print(net(x).shape)
