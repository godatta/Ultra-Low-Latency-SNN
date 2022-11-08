import torch
import torch.nn as nn
import math
from models.self_modules import HoyerBiAct


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG16_light_multi_steps(nn.Module):
    def __init__(self, T=2, leak=1.0, **kwargs):
        super(VGG16_light_multi_steps, self).__init__(**kwargs)
        self.T = T

        leak_dict   = {}
        for i, l in enumerate(self.features):
            if isinstance(l, nn.Conv2d):
                leak_dict['l'+str(i)] = nn.Parameter(torch.ones(self.features[l].out_channels)*leak)
        prev = len(self.features)
        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                leak_dict['l'+str(prev+i)] = nn.Parameter(torch.ones(self.features[l].out_channels)*leak)
        self.leak = nn.ParameterDict(leak_dict)

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width 		= x.size(2)
        self.height 	= x.size(3)			

        self.mem 	= {}
        self.mask 	= {}
        for i, l in enumerate(self.features):
            if isinstance(l, nn.Conv2d):
                self.mem[i] = torch.zeros(self.batch_size, l.out_channels, self.width, self.height).cuda()
                self.mask[i] = torch.ones(self.batch_size, l.out_channels, self.width, self.height).cuda()
            elif isinstance(l, nn.MaxPool2d):
                self.width = self.width//self.features[l].kernel_size
                self.height = self.height//self.features[l].kernel_size

        prev = len(self.features)

        for l in range(len(self.classifier)):

            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features).cuda()
                self.mask[prev+l] = torch.ones(self.mem[prev+l].shape).cuda()

    def forward(self, x):
        self.neuron_init(x)
        for i in range(self.T):
            out_prev = x

            for j, l in enumerate(self.features):
                if isinstance(l, nn.Conv2d):
                    out_prev = l(out_prev)
                    self.mem[j] = getattr(self.leak, 'l'+str(j)) * self.mem[j] + out_prev
                    out_prev = self.mem[j]
                elif isinstance(l, (nn.MaxPool2d, HoyerBiAct, nn.Dropout)):
                    out_prev = l(out_prev)
            prev = len(self.features)
            cls_len = len(self.classifier)
            out_prev = out_prev.view(out_prev.size(0), -1)
            for j, l in enumerate(self.classifier):
                if isinstance(l, nn.Linear):
                    out_prev = l(out_prev)
                    if j == cls_len-1:
                        self.mem[prev+j] = self.mem[prev+j] + out_prev
                    else:
                        self.mem[prev+j] = getattr(self.leak, 'l'+str(prev+j)) * self.mem[prev+j] + out_prev
                        out_prev = self.mem[len(self.features)+j]
                elif isinstance(l, (HoyerBiAct, nn.Dropout)):
                    out_prev = l(out_prev)
        return self.mem[prev+cls_len-1]      
