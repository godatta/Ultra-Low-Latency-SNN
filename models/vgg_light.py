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


class VGG16_light(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=1000, dataset = 'IMAGENET', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', loss_type='sum', spike_type = 'sum', bn_type='bn', start_spike_layer=0, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=0, im_size=224, if_set_0=True):
        super(VGG16_light, self).__init__()
        self.dataset = dataset
        self.spike_type = spike_type
        self.x_thr_scale = x_thr_scale
        self.if_spike = True
        self.conv_dropout = conv_dropout
        self.if_set_0 = if_set_0
        self.features = self._make_layers(cfg[vgg_name])
        self.loss_type = loss_type
        fc_spike_type = 'fixed' if spike_type == 'fixed' else 'sum'
        if dataset=='IMAGENET':
            self.classifier = nn.Sequential(
                            nn.Linear((im_size//32)**2*512, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, labels, bias=False)
            )
        elif dataset=='CIFAR10':
            self.classifier = nn.Sequential(
                            nn.Linear(2048, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, 4096, bias=False),
                            HoyerBiAct(spike_type=fc_spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                            nn.Dropout(linear_dropout),
                            nn.Linear(4096, labels, bias=False))
        # self._initialize_weights2()
                            
    def hoyer_loss(self, x, thr=0.0):
        # return torch.sum(x)
        x[x<0.0] = 0
        x[x>=thr] = thr
        if torch.sum(torch.abs(x))>0: #  and l < self.start_spike_layer
            return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
            # if self.loss_type == 'mean':
            #     return torch.mean(torch.sum(torch.abs(x), dim=(1,2,3))**2 / torch.sum((x)**2, dim=(1,2,3)))
            # elif self.loss_type == 'sum':
            #     return  (torch.sum(torch.abs(x))**2 / torch.sum((x)**2))
            # elif self.loss_type == 'cw':
            #     hoyer_thr = torch.sum((x)**2, dim=(0,2,3)) / torch.sum(torch.abs(x), dim=(0,2,3))
            #     # 1.0 is the max thr
            #     hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
            #     return torch.mean(hoyer_thr)
        return 0.0
    def forward(self, x):
        act_loss = 0.0
        out = x
        for l in self.features:
            # out = l(out)
            if isinstance(l, HoyerBiAct):
                # act_loss += self.hoyer_loss(out.clone())
                act_loss += self.hoyer_loss(out.clone(), l.threshold.clone().detach())
            out = l(out)
        
        out = out.view(out.size(0), -1)
        
        for i,l in enumerate(self.classifier):
            # out = l(out)
            if isinstance(l, HoyerBiAct):
                # act_loss += self.hoyer_loss(out.clone())
                act_loss += self.hoyer_loss(out.clone(), l.threshold.clone().detach())
            out = l(out)
 
        return out, act_loss

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        if self.dataset == 'IMAGENET':
            cfg.append('M')
        for i,x in enumerate(cfg):
            
            if x == 'M':
                continue
            conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1, bias=False)

            if i+1 < len(cfg) and cfg[i+1] == 'M':
                layers += [
                        conv,
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(x),
                        HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                        nn.Dropout(self.conv_dropout)]
                # layers += [
                #         conv,
                #         nn.BatchNorm2d(x),
                #         HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                #         nn.Dropout(self.conv_dropout),
                #         nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                        conv,
                        nn.BatchNorm2d(x),
                        HoyerBiAct(num_features=x, spike_type=self.spike_type, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike, if_set_0=self.if_set_0),
                        nn.Dropout(self.conv_dropout)]
            in_channels = x
        return nn.Sequential(*layers)
    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
