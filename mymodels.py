import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

class ConvBnReluLayer(nn.Module):
    def __init__(self, nfilter_in, nfilter_out):
        super(ConvBnReluLayer, self).__init__()

        self.conv = nn.Conv2d(nfilter_in, nfilter_out, 3, padding = 1)
        self.bn2d = nn.BatchNorm2d(nfilter_out, affine=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn2d(x)
        x = F.relu(x)
        return x

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

    def convbnlayers(self):
        for i in range(self.nlayers):
            yield getattr(self, "convbnrelu"+str(i))

    def named_actgrad(self, prefix=""):
        for (i, layer) in enumerate(self.convbnlayers()):
            yield (prefix+"convbnlayer"+str(i), layer.gradact)

    def keep_actgrad(self):
        def fun_hook(module, grad_input, grad_output):
            module.gradact += grad_output[0].pow(2).sum(-1).sum(-1).mean(dim=0)

        for layer in self.convbnlayers():
            layer.register_backward_hook(fun_hook)

    def reset_actgrad(self):
        for layer in self.convbnlayers():
            layer.gradact = 0.

class VGGNet(nn.Module):
    def __init__(self, K=1):
        super(VGGNet, self).__init__()
        self.vgglayer1 = VGGLayer(2, 3, K*64, 0.3)
        self.vgglayer2 = VGGLayer(3, K*64, K*128, 0.4)
        self.vgglayer3 = VGGLayer(3, K*128, K*256, 0.4)
        self.vgglayer4 = VGGLayer(3, K*256, K*512, 0.4)
        self.vgglayer5 = VGGLayer(3, K*512, K*512, 0.4)

        self.fc1 = nn.Linear(K*512, K*512)
        self.bn = nn.BatchNorm1d(K*512, affine=False)
        #self.fc2 = nn.Linear(512,10)

    def forward(self, x):
        for _, layer in self.named_vgglayers():
            x = layer(x)

        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.dropout(x,p = 0.5, training=self.training)
        return x

    def named_vgglayers(self, prefix=""):
        for i in range(1,6):
            name = "vgglayer{}".format(i)
            yield name, getattr(self, name)

    def named_actgrad(self, prefix=""):
        for namevgg, layer in self.named_vgglayers():
            for name, agrad in layer.named_actgrad(prefix=prefix+"."+namevgg+"."):
                yield name, agrad
        yield (prefix+".fc1", self.fc1.gradact)

    def keep_actgrad(self):
        for _, layer in self.named_vgglayers():
            layer.keep_actgrad()

        def fun_hook_linear(module, grad_input, grad_output):
            module.gradact += grad_output[0].pow(2).mean(dim=0)
        self.fc1.register_backward_hook(fun_hook_linear)

    def reset_actgrad(self):
        for _, layer in self.named_vgglayers():
            layer.reset_actgrad()
        self.fc1.gradact = 0.


class LinearClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        #x = F.log_softmax(x, dim=1)
        x = F.log_softmax(x, dim=1)
        return x

class LinearClassifierRNN(nn.Module):
    def __init__(self, nhid, ntoken, tie_weights = False, encoder = None, ninp = -1):
        super(LinearClassifierRNN, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.ntoken = ntoken

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = encoder.weight

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, output):
        ret = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return F.log_softmax(ret, dim = 1)

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
