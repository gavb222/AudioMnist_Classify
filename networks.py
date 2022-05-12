import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, activation = True, pooling = True):
        super(ConvRelu1d, self).__init__()
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding = self.padding)
        self.activation = activation
        self.pooling = pooling
        if self.activation:
            self.relu = nn.ReLU(inplace = True)
        if self.pooling:
            self.maxpool = nn.MaxPool1d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.relu(x)
        if self.pooling:
            x = self.maxpool(x)
        return x

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size, activation = True):
        super(FCLayer, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.relu(x)
        return x

class AudioNet(nn.Module):
    def __init__(self, in_length, n_classes):
        super(AudioNet, self).__init__()
        self.conv1 = ConvRelu1d(1,100)
        self.conv2 = ConvRelu1d(100,64)
        self.conv3 = ConvRelu1d(64,128)
        self.conv4 = ConvRelu1d(128,128)
        self.conv5 = ConvRelu1d(128,128)
        #self.conv6 = ConvRelu1d(128,128)
        self.linear1 = FCLayer(in_length * 4, 1024)
        self.linear2 = FCLayer(1024,512)
        self.linear3 = FCLayer(512, n_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        x = torch.flatten(x,start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
