import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReluLayer(nn.Module):
    def __init__(self, nfilter_in, nfilter_out):
        super(ConvBnReluLayer, self).__init__()
        
        self.conv = nn.Conv2d(nfilter_in, nfilter_out, 3, padding = 1)
        self.bn2d = nn.BatchNorm2d(nfilter_out)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn2d(x)
        x = F.relu(x)
        return x

    def loss(self):
        return self.conv.weight.pow(2).sum()
    

class VGGLayer(nn.Module):
    def __init__(self, nlayers, nfilter_in, nfilter_out, p):
        super(VGGLayer, self).__init__()
        self.nlayers = nlayers
        self.p = p
        
        self.convbnrelu0 = ConvBnReluLayer(nfilter_in, nfilter_out)
        for i in range(1, nlayers):
            setattr(self, "convbnrelu"+str(i),ConvBnReluLayer(nfilter_out, nfilter_out))
        
    def forward(self, x):
        for i in range(self.nlayers - 1):
            convbnrelulayer = getattr(self, "convbnrelu"+str(i))
            x = convbnrelulayer(x)
            x = F.dropout2d(x, p=self.p, training=self.training)
        convbnrelulayer = getattr(self, "convbnrelu"+str(self.nlayers-1))
        x = convbnrelulayer(x)
        x = F.max_pool2d(x, 2, stride = 2)
        return x

    def loss(self):
        ret = 0
        for i in range(self.nlayers):
            convbnrelulayer = getattr(self, "convbnrelu"+str(i))
            ret += convbnrelu.loss()
        return ret
    

class VGGNet(nn.Module):
    def __init__(self, K=1):
        super(VGGNet, self).__init__()
        self.vgglayer1 = VGGLayer(2, 3, K*64, 0.3)
        self.vgglayer2 = VGGLayer(3, K*64, K*128, 0.4)
        self.vgglayer3 = VGGLayer(3, K*128, K*256, 0.4)
        self.vgglayer4 = VGGLayer(3, K*256, K*512, 0.4)
        self.vgglayer5 = VGGLayer(3, K*512, K*512, 0.4)

        self.fc1 = nn.Linear(K*512, K*512)
        self.bn = nn.BatchNorm1d(K*512)
        #self.fc2 = nn.Linear(512,10)

    def forward(self, x):
        x = self.vgglayer1(x)
        x = self.vgglayer2(x)
        x = self.vgglayer3(x)
        x = self.vgglayer4(x)
        x = self.vgglayer5(x)
        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.dropout(x,p = 0.5, training=self.training)
        return x

        
class LinearClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        #x = F.log_softmax(x, dim=1)
        x = F.softmax(x, dim=1)
        return x

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out
