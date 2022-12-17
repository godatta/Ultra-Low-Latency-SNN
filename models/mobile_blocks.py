import torch
import torch.nn as nn
import torch.nn.functional as F

from models.self_modules import HoyerBiAct_multi_step


class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        

    def forward(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

class HoyerBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False, spike_type='sum', x_thr_scale=1.0):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(HoyerBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 
        self.act_loss = 0.0

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        self.act1 = HoyerBiAct_multi_step(num_features=input_channel, spike_type=spike_type, x_thr_scale=x_thr_scale)
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.act2 = HoyerBiAct_multi_step(num_features=c, spike_type=spike_type, x_thr_scale=x_thr_scale)
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.act3 = HoyerBiAct_multi_step(num_features=c, spike_type=spike_type, x_thr_scale=x_thr_scale)
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def hoyer_loss(self, x, thr=0.0):
        # return torch.sum(x)
        x[x<0.0] = 0
        x[x>=thr] = thr
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))   

    def forward(self, inputs, timestep=1):
        # main path
        self.act_loss = self.hoyer_loss(inputs, self.act1.threshold.clone().detach())
        x = self.bn1(self.conv1(self.act1(inputs, timestep)))
        self.act_loss += self.hoyer_loss(x, self.act2.threshold.clone().detach())
        x = self.bn2(self.conv2(self.act2(x, timestep)))
        self.act_loss += self.hoyer_loss(x, self.act3.threshold.clone().detach())
        x = self.bn3(self.conv3(self.act3(x, timestep)))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)

    print(x.shape)
    BaseBlock.alpha = 0.5
    b = BaseBlock(6, 5, downsample = True)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())