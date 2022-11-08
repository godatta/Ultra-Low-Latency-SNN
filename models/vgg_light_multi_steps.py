import torch
import torch.nn as nn
import math
from models.self_modules import HoyerBiAct
from models.vgg_light import VGG16_light


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG16_light_multi_steps(VGG16_light):
    def __init__(self, T=2, leak=1.0, **kwargs):
        super(VGG16_light_multi_steps, self).__init__(**kwargs)
        self.T = T

        leak_dict   = {}
        count = 0
        for l in self.features:
            if isinstance(l, nn.Conv2d):
                leak_dict['l'+str(count)] = nn.Parameter(torch.tensor(leak))
                count += 1
        for l in self.classifier:
            if isinstance(l, nn.Linear):
                leak_dict['l'+str(count)] = nn.Parameter(torch.tensor(leak))
                count += 1
        self.leak = nn.ParameterDict(leak_dict)

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width 		= x.size(2)
        self.height 	= x.size(3)			

        self.mem 	= {}
        self.mask 	= {}
        count = 0
        for i, l in enumerate(self.features):

            if isinstance(l, nn.Conv2d):
                if isinstance(self.features[i+1], nn.MaxPool2d):
                    self.width = self.width//self.features[i+1].kernel_size
                    self.height = self.height//self.features[i+1].kernel_size
                self.mem[count] = torch.zeros(self.batch_size, l.out_channels, self.width, self.height).cuda()
                count += 1
                

        for l in self.classifier:
            if isinstance(l, nn.Linear):
                self.mem[count] = torch.zeros(self.batch_size, l.out_features).cuda()
                count += 1
        # for l in self.mem.keys():
        #     print(self.mem[l].shape)

    def forward(self, x):
        self.neuron_init(x)
        act_loss = 0.0
        for i in range(self.T):
            out = x
            count = 0
            for l in self.features:
                if isinstance(l, HoyerBiAct):
                    self.mem[count] = getattr(self.leak, 'l'+str(count)) * self.mem[count] + out
                    out = l(self.mem[count])
                    # reset mem
                    # print('count: {}, mem shape: {}, out shape: {}, coef shape: {}'.format(count, self.mem[count].shape, out.shape, (l.running_hoyer_thr[None, :, None, None]*(out>0)).shape))
                    self.mem[count] = self.mem[count] - l.threshold.clone().detach() * l.running_hoyer_thr[None, :, None, None] * (out>0).float()
                    act_loss += self.hoyer_loss(out.clone(), l.threshold.clone().detach())
                    count += 1
                else:
                    out = l(out)

            out = out.view(out.size(0), -1)
            for l in self.classifier[:-1]:
                if isinstance(l, nn.Linear):
                    out = l(out)
                elif isinstance(l, HoyerBiAct):
                    self.mem[count] = getattr(self.leak, 'l'+str(count)) * self.mem[count] + out
                    out = l(out)
                    # reset mem
                    self.mem[count] = self.mem[count] - l.threshold * l.running_hoyer_thr * (out>0).float()
                    act_loss += self.hoyer_loss(out.clone(), l.threshold.clone().detach())
                    count += 1
                else:
                    out = l(out)
            # print('count: {}, mem shape: {}, out shape: {}'.format(count, self.mem[count].shape, out.shape))
            self.mem[count] = self.mem[count] + self.classifier[-1](out)
        return self.mem[count], act_loss   
