import torch
import torch.nn as nn
import torch.nn.functional as F
from models.self_modules import HoyerBiAct_multi_step


__all__ = ['resnet18', 'resnet20', 'resnet34', 'resnet34_cifar', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(inplanes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(inplanes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, spike_type='sum', \
         epoch=1, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=True, if_set_0=False):
        super(BasicBlock, self).__init__()

        self.epoch = epoch
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale
        self.act = HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)

        self.conv = conv3x3(inplanes, planes,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, T):
        residual = x
        # always spike
        out = self.act(x, T)
        out = self.conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            # residual = self.downsample(x)
            for l in self.downsample:
                if isinstance(l, HoyerBiAct_multi_step):
                    residual = l(residual, T)
                else:
                    residual = l(residual)

        out += residual

        return out

class BasicBlock_double(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, spike_type='sum', \
         epoch=1, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=True, if_set_0=False):
        super(BasicBlock_double, self).__init__()


        self.epoch = epoch
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale

        self.bi_act1        = HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.bi_conv1       = conv3x3(inplanes, planes,stride=stride)
        self.bn1            = nn.BatchNorm2d(planes)
        self.bi_act2        = HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.bi_conv2       = conv3x3(planes, planes,stride=1)
        self.bn2            = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # always spike
        out = self.bi_act1(x)
        out = self.bi_conv1(out)
        out = self.bn1(out)
        out = self.bi_act2(out)
        out = self.bi_conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, spike_type='sum', x_thr_scale=1.0, if_spike=True, if_set_0=False):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # 64, 64, 1, 1
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = HoyerBiAct_multi_step(num_features=planes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False) # 64, 64, 3, 1
        
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.act3 = HoyerBiAct_multi_step(num_features=planes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            # self.downsample = downsample
            # 1. spike + conv(s=2) + bn
            # self.downsample = nn.Sequential(
            #     HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale),
            #     nn.Conv2d(inplanes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     # nn.BatchNorm2d(self.expansion*planes) # 08211953 without this line
            # )
            # 2.  maxpool + bn + spike + conv1x1 
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.BatchNorm2d(inplanes),
                HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0),
                conv1x1(inplanes, planes * self.expansion),
            )


    # def forward(self, x):
    #     # print(f'x.shape: {x.shape}')
    #     out = self.bn1(self.conv1(self.binary_act1(x)))
    #     out = self.bn2(self.conv2(self.binary_act2(out)))
    #     # out = self.bn3(self.conv3(self.binary_act3(out)))
    #     # out += self.downsample(x)
    #     out = self.conv3(self.binary_act3(out))
    #     out += self.downsample(x)
    #     out = self.bn3(out)
    #     return out
    # foward 2.0
    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        # x = self.bn1(x)
        # out = self.conv1(self.binary_act1(x))
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.conv3(self.act3(self.bn3(out)))
        out += self.downsample(x)
        return out
class Bottleneck_v2(nn.Module): # spike->conv
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, spike_type='sum', x_thr_scale=1.0, if_spike=True, if_set_0=False):
        super(Bottleneck_v2, self).__init__()
        self.act1 = HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # 64, 64, 1, 1
        self.bn1 = nn.BatchNorm2d(planes)

        self.act2 = HoyerBiAct_multi_step(num_features=planes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False) # 64, 64, 3, 1
        self.bn2 = nn.BatchNorm2d(planes)

        self.act3 = HoyerBiAct_multi_step(num_features=planes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            # self.downsample = downsample
            # 1. spike + conv(s=2) + bn
            # self.downsample = nn.Sequential(
            #     HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0),
            #     nn.Conv2d(inplanes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     # nn.BatchNorm2d(self.expansion*planes) # 08211953 without this line
            # )
            # 2.  maxpool + bn + spike + conv1x1 
            # self.downsample = nn.Sequential(
            #     nn.MaxPool2d(kernel_size=stride, stride=stride),
            #     nn.BatchNorm2d(inplanes),
            #     HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0),
            #     conv1x1(inplanes, planes * self.expansion),
            # )
            # 3.0 maxpooling->spike->conv->bn
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                HoyerBiAct_multi_step(num_features=inplanes, spike_type=spike_type, x_thr_scale=x_thr_scale, if_spike=if_spike, if_set_0=if_set_0),
                conv1x1(inplanes, planes * self.expansion),
                nn.BatchNorm2d(planes * self.expansion),
            )


    # def forward(self, x):
    #     # print(f'x.shape: {x.shape}')
    #     out = self.bn1(self.conv1(self.binary_act1(x)))
    #     out = self.bn2(self.conv2(self.binary_act2(out)))
    #     # out = self.bn3(self.conv3(self.binary_act3(out)))
    #     # out += self.downsample(x)
    #     out = self.conv3(self.binary_act3(out))
    #     out += self.downsample(x)
    #     out = self.bn3(out)
    #     return out
    # foward 2.0
    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        # x = self.bn1(x)
        # out = self.conv1(self.binary_act1(x))
        out = self.bn1(self.conv1(self.act1(x)))
        out = self.bn2(self.conv2(self.act2(out)))
        out = self.bn3(self.conv3(self.act3(out)))
        out += self.downsample(x)
        return out

class HoyerResNet_multi_steps(nn.Module):
    def __init__(self, block, num_blocks, labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', loss_type='sum', spike_type = 'sum', bn_type='bn', start_spike_layer=50, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=1, im_size=224, last_bn_c=512, if_set_0=False, T=1, leak=1.0):
        
        super(HoyerResNet_multi_steps, self).__init__()
        self.inplanes = 64
        self.spike_type     = spike_type
        self.loss_type      = loss_type
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = True if start_spike_layer == 0 else False 
        self.if_set_0       = if_set_0
        self.T              = T
        self.leak           = leak
        self.fc_leak        = nn.Parameter(torch.tensor(1.0))
        self.test_hoyer_thr = torch.tensor([0.0]*15)    
        if dataset == 'CIFAR10':
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # self.conv1 = customConv2(in_channels=3, out_channels=64, kernel_size=(3 ,3), stride = 1, padding = 1)
            self.conv1 = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                # customConv2(in_channels=3, out_channels=64, kernel_size=(3 ,3), stride = 1, padding = 1),
                                nn.BatchNorm2d(64),
                                HoyerBiAct_multi_step(num_features=64, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),

                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),

                                HoyerBiAct_multi_step(num_features=64, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                # nn.MaxPool2d(2),
                                # nn.BatchNorm2d(64),
                                # HoyerBiAct_multi_step(num_features=64, spike_type=self.spike_type, if_set_0=self.if_set_0)
                                )
        elif dataset == 'IMAGENET':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        else:
            raise RuntimeError('only for ciafar10 and imagenet now')
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.bn1     = nn.BatchNorm2d(last_bn_c)
        self.fc_act = HoyerBiAct_multi_step(spike_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, labels)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 1.0 maxpool + bn + spike + conv1x1 for resnet18 with vgg, it is the best
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(self.inplanes),
                HoyerBiAct_multi_step(num_features=self.inplanes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                # nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale))

        return nn.Sequential(*layers)

    def hoyer_loss(self, x):
        x[x<0]=0
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            if self.loss_type == 'mean':
                return torch.mean(torch.sum(torch.abs(x), dim=(1,2,3))**2 / torch.sum((x)**2, dim=(1,2,3)))
            elif self.loss_type == 'sum':
                return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
            elif self.loss_type == 'cw':
                hoyer_thr = torch.sum((x)**2, dim=(0,2,3)) / torch.sum(torch.abs(x), dim=(0,2,3))
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                return torch.mean(hoyer_thr)
        return 0.0


    def forward(self, x):
        self.final_out = 0.0
        act_out = 0.0
        for T in range(self.T):
            prev_x = x
            prev_x = self.conv1(prev_x)
            prev_x = self.maxpool(prev_x)
            prev_x = self.bn1(prev_x) #  for 1.0
            # act_out += self.hoyer_loss(prev_x.clone())

            for i,layers in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for l in layers:
                    # print(i, prev_x.shape)
                    # if isinstance(l, HoyerBiAct_multi_step):
                    #     prev_x = l(prev_x, T+1)    
                    # else:
                    #     prev_x = l(prev_x)
                    prev_x = l(prev_x, T+1) 
                    # act_out += self.hoyer_loss(prev_x.clone())
                # prev_x = layers(prev_x)
                # act_out += self.hoyer_loss(prev_x.clone())
            # prev_x = self.bn1(prev_x) # for 2.0
            prev_x = self.avgpool(prev_x)
            prev_x = prev_x.view(prev_x.size(0), -1)
            # act_out += self.hoyer_loss(prev_x.clone())
            prev_x = self.fc_act(prev_x, T+1)
            self.final_out = self.final_out*self.fc_leak + self.fc(prev_x)

        return self.final_out, act_out


def resnet18_multi_steps(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model. """
    model = HoyerResNet_multi_steps(BasicBlock, [4, 4, 4, 4],last_bn_c=512, **kwargs)
    return model

def resnet20_multi_steps(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model. """
    model = HoyerResNet_multi_steps(BasicBlock, [4, 4, 4, 4], **kwargs)
    # model = HoyerResNet(BasicBlock_double, [2,2,2,2], **kwargs)
    return model

def resnet34_cifar(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model. """
    model = HoyerResNet_multi_steps(BasicBlock, [6, 8, 10, 6], **kwargs)
    return model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model. """
    model = HoyerResNet_multi_steps(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


