# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.self_modules import HoyerBiAct_multi_step
from models.mobile_blocks import BaseBlock, HoyerBlock, conv_dw


class MobileNetV2Cifar(nn.Module):
    def __init__(self, alpha = 1, labels=1000, dataset = 'IMAGENET', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', loss_type='sum', spike_type = 'sum', bn_type='bn', start_spike_layer=0, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=0, im_size=224):
        super(MobileNetV2Cifar, self).__init__()
        self.labels = labels

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, labels)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        act_loss = 0.0
        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x,act_loss

class HoyerMobileNetV2Cifar(nn.Module):
    def __init__(self, alpha = 1, labels=1000, dataset = 'IMAGENET', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', loss_type='sum', spike_type = 'sum', bn_type='bn', start_spike_layer=0, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=0, im_size=224, if_set_0=False, T=1, leak=1.0):
        super(HoyerMobileNetV2Cifar, self).__init__()
        self.labels = labels
        self.if_set_0 = if_set_0
        self.T = T
        self.leak = leak
        self.fc_leak        = nn.Parameter(torch.tensor(leak))

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        HoyerBlock.alpha = alpha
        self.bottlenecks = nn.ModuleList([
            HoyerBlock(32, 16, t = 1, downsample = False, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(16, 24, downsample = False, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(24, 24, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(24, 32, downsample = False, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(32, 32, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(32, 32, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(32, 64, downsample = True, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(64, 64, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(64, 64, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(64, 64, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(64, 96, downsample = False, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(96, 96, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(96, 96, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(96, 160, downsample = True, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(160, 160, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(160, 160, spike_type=spike_type, x_thr_scale=x_thr_scale),
            HoyerBlock(160, 320, downsample = False, spike_type=spike_type, x_thr_scale=x_thr_scale)])

        # last conv layers and fc layer
        self.conv1_act = HoyerBiAct_multi_step(num_features=int(320*alpha), spike_type=spike_type, x_thr_scale=x_thr_scale)
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        fc_spike_type = 'fixed' if spike_type == 'fixed' else 'sum'
        self.fc_act = HoyerBiAct_multi_step(num_features=1, spike_type=fc_spike_type, x_thr_scale=x_thr_scale)
        self.fc = nn.Linear(1280, labels)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def hoyer_loss(self, x, thr=0.0):
        # return torch.sum(x)
        x[x<0.0] = 0
        x[x>=thr] = thr
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))  
        else:
            return 0.0

    def forward(self, inputs):
        self.final_out = 0.0
        act_loss = 0.0
        for t in range(self.T):
            # first conv layer
            x = self.bn0(self.conv0(inputs))
            # assert x.shape[1:] == torch.Size([32, 32, 32])

            # bottlenecks
            for block in self.bottlenecks:
                x = block(x, t+1)
                act_loss += block.act_loss
            # assert x.shape[1:] == torch.Size([320, 8, 8])

            # last conv layer
            act_loss += self.hoyer_loss(x, self.conv1_act.threshold.clone().detach())
            x = self.conv1_act(x)
            x = self.bn1(self.conv1(x))
            # assert x.shape[1:] == torch.Size([1280,8,8])

            # global pooling and fc (in place of conv 1x1 in paper)
            x = F.adaptive_max_pool2d(x, 1)
            x = x.view(x.shape[0], -1)
            act_loss += self.hoyer_loss(x, self.fc_act.threshold.clone().detach())
            x = self.fc_act(x, t+1)
            self.final_out = self.final_out*self.fc_leak + self.fc(x)


        return self.final_out, act_loss

class HoyerMobileNetV1Cifar(nn.Module):
    def __init__(self, alpha = 1, labels=1000, dataset = 'IMAGENET', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', loss_type='sum', spike_type = 'sum', bn_type='bn', start_spike_layer=0, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=0, im_size=224, if_set_0=False, T=1, leak=1.0):
        super(HoyerMobileNetV1Cifar, self).__init__()
        self.T = T
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.model = nn.Sequential(
            conv_dw(32, 64, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(64, 128, 2, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(128, 128, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(128, 256, 2, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(256, 256, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(256, 512, 2, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 512, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 512, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 512, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 512, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 512, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(512, 1024, 2, spike_type=spike_type, x_thr_scale=x_thr_scale),
            conv_dw(1024, 1024, 1, spike_type=spike_type, x_thr_scale=x_thr_scale),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)
        self.fc_act = HoyerBiAct_multi_step(spike_type='sum', x_thr_scale=x_thr_scale)
        self.fc_leak = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        act_loss = 0.0
        final_out = 0.0
        for T in range(self.T):
            prev_x = x
            prev_x = self.bn1(self.conv1(prev_x))
            for conv_block in self.model:
                prev_x = conv_block(prev_x, T+1)
                act_loss += conv_block.act_loss
            prev_x = self.avgpool(prev_x)
            prev_x = prev_x.view(-1, 1024)
            prev_X = self.fc_act(prev_x)
            final_out = final_out*self.fc_leak + self.fc(prev_x)
        return final_out, act_loss / self.T

if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from count import measure_model
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    print(x.shape)

    net = HoyerMobileNetV2Cifar(10, alpha = 1)
    y = net(x)

    print(x.shape)
    print(y.shape)

    f, c = measure_model(net, 32, 32)
    print("model size %.4f M, ops %.4f M" %(c/1e6, f/1e6))

    # size = 1
    # for param in net.parameters():
    #     arr = np.array(param.size())
        
    #     s = 1
    #     for e in arr:
    #         s *= e
    #     size += s

    # print("all parameters %.2fM" %(size/1e6) )