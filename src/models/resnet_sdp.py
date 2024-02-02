import torch.nn as nn
from copy import deepcopy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        expansion = 1
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        _out = self.conv2(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        if get_features:
            if detached:
                d_out = self.conv1(x.detach())
                d_out = self.conv2(d_out)
                if self.downsample is not None:
                    d_shortcut = self.downsample(x.detach())
                else:
                    d_shortcut = x.detach()
                d_out = d_out + d_shortcut
                features.append(d_out)
            else:
                features.append(_out)
        return [_out, features, get_features, detached]


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv3 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.downsample = downsample

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        _out = self.conv2(_out)
        _out = self.conv3(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        if get_features:
            if detached:
                d_out = self.conv1(x.detach())
                d_out = self.conv2(d_out)
                d_out = self.conv3(d_out)
                if self.downsample is not None:
                    d_shortcut = self.downsample(x.detach())
                else:
                    d_shortcut = x.detach()
                d_out = d_out + d_shortcut
                features.append(d_out)
            else:
                features.append(_out)
        return [_out, features, get_features, detached]


class ResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(
                opt=opt,
                in_channels=inChannels,
                out_channels=outChannels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module(
            "block0", block(opt, inChannels, outChannels, stride, downsample)
        )
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module(
                "block{}".format(i), block(opt, inChannels, outChannels)
            )

    def forward(self, x, features=None, get_features=False, detached=False):
        return self.blocks([x, features, get_features, detached])[:2]


class ResNet(nn.Module):
    def __init__(self, opt):
        super(ResNet, self).__init__()
        depth = opt.depth
        if depth in [20, 32, 44, 56, 110, 1202]:
            blocktype, self.nettype = "BasicBlock", "cifar"
        elif depth in [164, 1001]:
            blocktype, self.nettype = "BottleneckBlock", "cifar"
        elif depth in [18, 34]:
            blocktype, self.nettype = "BasicBlock", "imagenet"
        elif depth in [50, 101, 152]:
            blocktype, self.nettype = "BottleneckBlock", "imagenet"
        assert depth in [20, 32, 44, 56, 110, 1202, 164, 1001, 18, 34, 50, 101, 152]

        if blocktype == "BasicBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 6 == 0, (
                "Depth should be 6n+2, and preferably one of 20, 32, 44, 56, 110, 1202"
            )
            n = (depth - 2) // 6
            block = BasicBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BottleneckBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 9 == 0, "Depth should be 9n+2, and preferably one of 164 or 1001"
            n = (depth - 2) // 9
            block = BottleneckBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BasicBlock" and self.nettype == "imagenet":
            assert depth in [18, 34]
            num_blocks = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
            block = BasicBlock
            in_planes, out_planes = 64, 512  # 20, 160
        elif blocktype == "BottleneckBlock" and self.nettype == "imagenet":
            assert depth in [50, 101, 152]
            if depth == 50:
                num_blocks = [3, 4, 6, 3]
            elif depth == 101:
                num_blocks = [3, 4, 23, 3]
            elif depth == 152:
                num_blocks = [3, 8, 36, 3]
            block = BottleneckBlock
            in_planes, out_planes = 64, 512
        else:
            assert 1 == 2

        self.num_classes = opt.num_classes
        self.initial = InitialBlock(
            opt=opt, out_channels=in_planes, kernel_size=3, stride=1, padding=1
        )
        if self.nettype == "cifar":
            self.group1 = ResidualBlock(opt, block, 16, 16, n, stride=1)
            self.group2 = ResidualBlock(
                opt, block, 16 * block.expansion, 32, n, stride=2
            )
            self.group3 = ResidualBlock(
                opt, block, 32 * block.expansion, 64, n, stride=2
            )
        elif self.nettype == "imagenet":
            self.group1 = ResidualBlock(
                opt, block, 64, 64, num_blocks[0], stride=1
            )  # For ResNet-S, convert this to 20,20
            self.group2 = ResidualBlock(
                opt, block, 64 * block.expansion, 128, num_blocks[1], stride=2
            )  # For ResNet-S, convert this to 20,40
            self.group3 = ResidualBlock(
                opt, block, 128 * block.expansion, 256, num_blocks[2], stride=2
            )  # For ResNet-S, convert this to 40,80
            self.group4 = ResidualBlock(
                opt, block, 256 * block.expansion, 512, num_blocks[3], stride=2
            )  # For ResNet-S, convert this to 80,160
        else:
            assert 1 == 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim_out = out_planes * block.expansion
        #self.fc = FinalBlock(opt=opt, in_channels=out_planes * block.expansion)
        self.fc = nn.Linear(out_planes * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, get_feature=False, get_features=False, detached=False):
        features = []
        out_init = self.initial(x)
        if get_features:
            features.append(out_init)
        out1, features = self.group1(out_init, features, get_features, detached)
        out2, features = self.group2(out1, features, get_features, detached)
        out3, features = self.group3(out2, features, get_features, detached)
        if self.nettype == "imagenet":
            out3, features = self.group4(out3, features, get_features, detached)
        feature = self.pool(out3)
        feature = feature.view(x.size(0), -1)
        out = self.fc(feature)
        if get_feature:
            return out, feature
        elif get_features:
            return out, features
        else:
            return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        opt,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

        layer = [conv]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                # layer = [st, bn]
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "2d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                # layer = [conv, st, bn]
                layer = [conv, bn]

        if opt.activetype is not "None":
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(conv)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


class FCBlock(nn.Module):
    def __init__(self, opt, in_channels, out_channels, bias=False):
        super(FCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels
        lin = nn.Linear(in_channels, out_channels, bias=bias)

        layer = [lin]
        if opt.bn:
            if opt.preact:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=in_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [bn]
            else:
                bn = getattr(nn, opt.normtype + "1d")(
                    num_features=out_channels, affine=opt.affine_bn, eps=opt.bn_eps
                )
                layer = [lin, bn]

        if opt.activetype is not "None":
            active = getattr(nn, opt.activetype)()
            layer.append(active)

        if opt.bn and opt.preact:
            layer.append(lin)

        self.block = nn.Sequential(*layer)

    def forward(self, input):
        return self.block.forward(input)


def FinalBlock(opt, in_channels, bias=False):
    out_channels = opt.num_classes
    opt = deepcopy(opt)
    if not opt.preact:
        opt.activetype = "None"
    return nn.Linear(in_channels, out_channels)


def InitialBlock(opt, out_channels, kernel_size, stride=1, padding=0, bias=False):
    in_channels = opt.in_channels
    opt = deepcopy(opt)
    return ConvBlock(
        opt=opt,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
