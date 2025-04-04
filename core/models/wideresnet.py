import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

class BasicBlock(nn.Module):
    """
    Implements a basic block module for WideResNets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        dropRate (float): dropout rate.
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    Implements a network block module for WideResnets.
    Arguments:
        nb_layers (int): number of layers.
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        block (BasicBlock): type of basic block to be used.
        stride (int): stride of convolution.
        dropRate (float): dropout rate.
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet model
    Arguments:
        depth (int): number of layers.
        num_classes (int): number of output classes.
        widen_factor (int): width factor.
        dropRate (float): dropout rate.
    """
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0,
                 mean: Union[Tuple[float, ...], float] = (0.5, 0.5, 0.5),
                 std: Union[Tuple[float, ...], float] = (0.5, 0.5, 0.5),
                 padding: int = 0,
                 num_input_channels: int = 3,
                 normalize=True,
                 ):
        super(WideResNet, self).__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.normalize = normalize
        self.padding = padding

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.feat_dim = self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def normalize_(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            x = (x - self.mean_cuda) / self.std_cuda
        else:
            x = (x - self.mean) / self.std
        return x

    def denormalize_(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            x = (x - self.mean_cuda) / self.std_cuda
            x = x * self.std_cuda + self.mean_cuda
        else:
            x = x * self.std + self.mean
        return x

    def forward(self, x, feats=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if self.normalize:
            x = self.normalize_(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if not feats:
            return self.fc(out)
        else:
            return out, self.fc(out)

    def rf_output(self, x, intermediate_propagate=0, pop=0):
        if intermediate_propagate == 0:
            if self.padding > 0:
                x = F.pad(x, (self.padding,) * 4)
            if self.normalize:
                x = self.normalize_(x)
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            if pop == 1:
                return out
            out = self.block2(out)
            if pop == 2:
                return out
            out = self.block3(out)
            out = self.bn1(out)
            if pop == 3:
                return out
            out = self.relu(out)
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 1:
            out = x
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 2:
            out = x
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

        elif intermediate_propagate == 3:
            out = x
            out = self.relu(out)
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(-1, self.nChannels)
            return self.fc(out)

    
def wideresnet(name, logger, num_classes=10, normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device='cpu'):
    """
    Returns suitable Wideresnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
    Returns:
        torch.nn.Module.
    """
    name_parts = name.split('-')
    depth = int(name_parts[1])
    widen = int(name_parts[2])

    if normalize:
        if logger is not None:
            logger.log(f'WideResNet-{depth}-{widen} uses normalization {mean}, {std}.')
        return WideResNet(num_classes=num_classes, depth=depth, widen_factor=widen,
                          mean=mean, std=std, normalize=True)
    else:
        if logger is not None:
            logger.log(f'WideResNet-{depth}-{widen}.')
        return WideResNet(num_classes=num_classes, depth=depth, widen_factor=widen,
                          normalize=False)

    # return WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen)
