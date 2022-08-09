import torch
import torch.nn as nn
import torch.nn.functional as F
from models.self_modules import HoyerBiAct



__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, threshold=1.0,\
         epoch=1, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=0):
        super(BasicBlock, self).__init__()

        # self.binary_activation = nn.ReLU(inplace=True)
        # BinaryActivation()
        # self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.epoch = epoch
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale
        self.x_thr_scale    = x_thr_scale
        self.threshold      = threshold
        self.binary_activation = HoyerBiAct(num_features=inplanes, hoyer_type='sum', x_thr_scale=self.x_thr_scale, if_spike=if_spike)

        self.binary_conv = conv3x3(inplanes, planes,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # always spike
        out = self.binary_activation(x)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class ResNet_hoyer(nn.Module):
    def __init__(self, block, num_blocks, labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', hoyer_type='mean', act_mode = 'mean', bn_type='bn', start_spike_layer=50, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=1, im_size=224):
        
        super(ResNet_hoyer, self).__init__()
        self.inplanes = 64
        self.conv_dropout   = conv_dropout
        self.act_mode       = act_mode
        self.hoyer_type     = hoyer_type
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = True if start_spike_layer == 0 else False 
        if dataset == 'CIFAR10':
            self.pre_process = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                HoyerBiAct(num_features=64, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                                # nn.Dropout(self.conv_dropout),

                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                HoyerBiAct(num_features=64, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                                # nn.Dropout(self.conv_dropout),

                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                # nn.MaxPool2d(2),
                                # nn.BatchNorm2d(64),
                                # HoyerBiAct(num_features=64, hoyer_type=self.act_mode)
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, labels)
        self.relu_batch_num = 0


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.BatchNorm2d(self.inplanes),
                HoyerBiAct(num_features=self.inplanes, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                # nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def hoyer_loss(self, x, act_out):
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            if self.hoyer_type == 'mean':
                return torch.mean(torch.sum(torch.abs(x), dim=(1,2,3))**2 / torch.sum((x)**2, dim=(1,2,3))).clone()
            elif self.hoyer_type == 'sum':
                return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2)).clone()
            elif self.hoyer_type == 'cw':
                hoyer_thr = torch.sum((x)**2, dim=(0,2,3)) / torch.sum(torch.abs(x), dim=(0,2,3)).clone()
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                return torch.mean(hoyer_thr)
    def num_relu(self, x, min_thr_scale, max_thr_scale, thr):
        # epoch = 1
        # min = (x<epoch*1e-3).sum()
        # max = (x>1.0 - epoch*1e-3).sum()

        min = (x<=min_thr_scale*thr).sum()
        max = (x>=max_thr_scale*thr).sum()
        total = x.view(-1).shape[0]
        return torch.tensor([min, total-min-max, max, total])

    def forward(self, x):
        relu_batch_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        act_out = 0.0
        x = self.pre_process(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        act_out += self.hoyer_loss(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            act_out += self.hoyer_loss(x)
            self.relu_batch_num += self.num_relu(x, 0.0, 1.0, torch.max(x).clone().detach())
        


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x,[], relu_batch_num, act_out


def resnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = ResNet_hoyer(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

def resnet20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = ResNet_hoyer(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model

def resnet34_cifar(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = ResNet_hoyer(BasicBlock, [6, 8, 10, 6], **kwargs)
    return model

def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = ResNet_hoyer(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model
