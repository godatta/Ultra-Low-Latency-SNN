from tkinter.messagebox import NO
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from models.dynamic_conv import Dynamic_conv2d
from models.self_modules import HoyerBiAct, tdBatchNorm, ThrBiAct, SubBiAct
# from dynamic_conv import Dynamic_conv2d


cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512], # 13 conv + 3 fc
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class VGG_TUNABLE_THRESHOLD_tdbn(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', hoyer_type='mean', act_mode = 'mean', bn_type='bn', start_spike_layer=50, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max'):
        super(VGG_TUNABLE_THRESHOLD_tdbn, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.bn_type        = bn_type
        self.conv_type      = conv_type
        self.pool_pos       = pool_pos
        self.act_mode       = act_mode
        self.sub_act_mask   = sub_act_mask
        self.hoyer_type     = hoyer_type
        self.start_spike_layer = start_spike_layer
        self.x_thr_scale    = x_thr_scale
        self.pooling        = nn.MaxPool2d(kernel_size=2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)
        self.features       = self._make_layers(cfg[vgg_name])
        self.dropout_conv   = nn.Dropout(conv_dropout)
        self.dropout_linear = nn.Dropout(linear_dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.test_hoyer_thr = torch.tensor([0.0]*15)
        self.threshold_out  = []
        self.relu_batch_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
        self.threshold 	= {}
        self.perc = {}
        self.dis = []
        self.layer_output = {}
        # with open('output/ann_min_scale_vgg16_cifar10_0.3', 'rb') as f:
        #     self.min_thr_scale = torch.load(f)
        # with open('output/ann_max_scale_vgg16_cifar10_0.7', 'rb') as f:
        #     self.max_thr_scale = torch.load(f) 
        # with open('output/ann_min_scale_vgg16_cifar10_0.2', 'rb') as f:
        #     self.min_thr_scale = torch.load(f)
        # with open('output/ann_max_scale_vgg16_cifar10_0.8', 'rb') as f:
        #     self.max_thr_scale = torch.load(f)
        # self.scale_factor = torch.load('output/my_x_scale_factor')

        if net_mode == 'ori':
            self.min_thr_scale = [0.0]*15
            self.max_thr_scale = [1.0]*15
        elif net_mode[:3] == 'cut':
            temp_num = int(net_mode[-1:])
            with open('output/ann_min_scale_vgg16_cifar10_tdbn_0.{}'.format(temp_num), 'rb') as f:
                self.min_thr_scale = torch.load(f)
            with open('output/ann_max_scale_vgg16_cifar10_tdbn_0.{}'.format(10-temp_num), 'rb') as f:
                self.max_thr_scale = torch.load(f) 
        
        # define 3 fc
        if vgg_name == 'VGG6' and dataset!= 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*4*4, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name == 'VGG4' and dataset== 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 1024, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            #nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(1024, labels, bias=False)
                            )
        elif vgg_name!='VGG6' and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(2048, 4096, bias=True),
                            # SubBiAct(act_mode=self.act_mode, bit=1, act_size=(4096)) if self.sub_act_mask else ThrBiAct(act_mode=self.act_mode) ,
                            HoyerBiAct(hoyer_type='sum'),
                            # ThrBiAct(act_mode=self.act_mode) ,
                            # nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=True),
                            # SubBiAct(act_mode=self.act_mode, bit=1, act_size=(4096)) if self.sub_act_mask else ThrBiAct(act_mode=self.act_mode) ,
                            # ThrBiAct(act_mode=self.act_mode) ,
                            HoyerBiAct(hoyer_type='sum'),
                            # nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=True)
                            )
        elif vgg_name == 'VGG6' and dataset == 'MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(128*7*7, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        elif vgg_name!='VGG6' and dataset =='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(512*1*1, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
                            )
        for l in range(len(self.features)):
            if isinstance(self.features[l], (ThrBiAct, SubBiAct, HoyerBiAct)):
                self.threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(l)]  = nn.Parameter(torch.ones(9))
        prev = len(self.features)
        for l in range(len(self.classifier)-1):#-1
            if isinstance(self.classifier[l], (ThrBiAct, SubBiAct, HoyerBiAct)):
                self.threshold['t'+str(prev+l)]	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(prev+l)]  = nn.Parameter(torch.ones(9))
        # print(self.features)
        self.threshold 	= nn.ParameterDict(self.threshold)
        #self.epoch = nn.parameter(epoch)
        #self.epoch.requires_grad = False
        #self.cur = nn.ParameterDict(percentile)
        #self.cur.requires_grad = False
        
        self._initialize_weights2()

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result
    '''
    def relu_threshold (self, x, epoch, threshold):

        res_1 = (x<epoch*1e-3)*0
        res_2 = (epoch*1e-3<=x & x<threshold-epoch*1e-3)*((threshold/(threshold-epoch*2e-3))*x - (threshold*(epoch*1e-3)/(threshold-epoch*2e-3)))
        res_3 = (x>=threshold-epoch*1e-3)*(threshold)
        #out[out<epoch*1e-3] = 0
        #out[out>=epoch*1e-3] = (threshold/(threshold-epoch*2e-3)) - (threshold*(epoch*1e-3)/(threshold-epoch*2e-3))
        #out[out>=threshold-epoch*1e-3] = threshold

        return res_1 + res_2 + res_3
    '''    
    def num_relu(self, x, min_thr_scale, max_thr_scale, thr):
        # epoch = 1
        # min = (x<epoch*1e-3).sum()
        # max = (x>1.0 - epoch*1e-3).sum()

        min = (x<=min_thr_scale*thr).sum()
        max = (x>=max_thr_scale*thr).sum()
        total = x.view(-1).shape[0]
        return torch.tensor([min, total-min-max, max, total])
    
    def update_temperature(self):
        for l in range(len(self.features)):
            if isinstance(self.features[l], Dynamic_conv2d):
                (self.features[l]).update_temperature()
    
    def get_hoyer_thr(self, input, layer_index):
        
        out = torch.clamp(input, min=0.0, max=1.0)
        if layer_index >= 44:
            # hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
            hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
        else:
            # hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
            hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
            # hoyer_sum = torch.sum((out)**2, dim=(0,2,3)) / torch.sum(torch.abs(out), dim=(0,2,3))
            
            # hoyer_cw = torch.sum((out)**2, dim=(2,3)) / torch.sum(torch.abs(out), dim=(2,3))
            # hoyer_cw = torch.nan_to_num(hoyer_cw, nan=0.0)
            # hoyer_cw = torch.mean(hoyer_cw, dim=0)
            # N,C,W,H = input.shape
            # hoyer_thr = (torch.permute(hoyer_cw*(torch.ones(N,W,H,C)).cuda(), (0,3,1,2)))
        return hoyer_thr

    
    def forward(self, x, epoch=1):   #######epoch
        out_prev = x
        self.threshold_out = []
        self.relu_batch_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if epoch < 0:
            act_out = {}
        else:
            act_out = 0.0
        i = 0
        for l in range(len(self.features)):
            if isinstance(self.features[l], (nn.Conv2d, nn.MaxPool2d, Dynamic_conv2d, nn.AvgPool2d)):
                # print('layer: {}, shape: {}'.format(l, out_prev.shape))
                out_prev = self.features[l](out_prev)
            if isinstance(self.features[l], (nn.BatchNorm2d)):
                # out_prev = self.features[l](out_prev)
                if self.bn_type == 'bn':
                    out_prev = self.features[l](out_prev)
                elif self.bn_type == 'tdbn':
                    out_prev = self.features[l](out_prev, getattr(self.threshold, 't'+str(l+1)))
                elif self.bn_type == 'fake':
                    out_prev = self.features[l](out_prev, 0.25)
            if isinstance(self.features[l], (ThrBiAct, SubBiAct, HoyerBiAct)):
                # print('{}, shape: {}'.format(l, out_prev.shape))
                # 1. relu:
                # out = self.relu(out_prev)
                # 2. x/thr -> act
                # out = out_prev
                out = out_prev/getattr(self.threshold, 't'+str(l))
                # out = out_prev/torch.abs(getattr(self.threshold, 't'+str(l)))
                self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), l)
                out = self.features[l](out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], self.x_thr_scale, l, self.start_spike_layer)

                # 3. x/hoyer_thr -> act
                
                # hoyer_cw = torch.sum((out)**2, dim=(2,3)) / torch.sum(torch.abs(out), dim=(2,3))
                # hoyer_cw = torch.nan_to_num(hoyer_cw, nan=0.0)
                # hoyer_cw = torch.mean(hoyer_cw, dim=0)
                # N,C,W,H = out.shape
                # hoyer_thr = (torch.permute(hoyer_cw*(torch.ones(N,W,H,C)).cuda(), (0,3,1,2)))
                # out = self.relu(out_prev)
                # self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), l)
                # # hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
                # hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
                # out = out/hoyer_thr
                # out = self.act_func(out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], l, self.start_spike_layer)

                # 4.
                # out = out_prev/torch.abs(getattr(self.threshold, 't'+str(l)))
                # out = self.relu(out)
                # out = torch.clamp(out, max=1.0)
                # self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), l)
                # # out /= 0.618*(torch.sum((out.clone())**2) / torch.sum(torch.abs(out.clone())))
                # out /= (torch.sum((out.clone())**2) / torch.sum(torch.abs(out.clone())))
                # out = self.features[l](out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], l, self.start_spike_layer)

                
                self.relu_batch_num += self.num_relu(out.clone().detach(), 0.0, 1.0, torch.max(out).clone().detach())
                if epoch == -1:
                    act_out[l] = out.clone().detach()
                # elif epoch == -2:
                #     act_out[l] = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3))) # hoyer threshold
                else:
                    if torch.sum(torch.abs(out))>0: #  and l < self.start_spike_layer
                        if self.hoyer_type == 'mean':
                            act_out += torch.mean(torch.sum(torch.abs(out), dim=(1,2,3))**2 / torch.sum((out)**2, dim=(1,2,3))).clone()
                        elif self.hoyer_type == 'sum':
                            act_out +=  (torch.sum(torch.abs(out))**2 / torch.sum((out)**2)).clone()
                        # elif self.hoyer_type == 'mask':
                        #     mask = torch.zeros_like(out).cuda()
                        #     mask[out<torch.max(out).clone().detach()] = 1.0
                        #     if torch.sum(torch.abs(out*mask))>0:
                        #         act_out += (torch.mean(torch.sum(torch.abs(out*mask), dim=(1,2,3))**2 / torch.sum((out*mask)**2, dim=(1,2,3)))).clone()
                        # elif self.hoyer_type == 'l1':
                        #     mask = torch.zeros_like(out).cuda()
                        #     mask[out<torch.max(out).clone().detach()] = 1.0
                        #     act_out += torch.sum(torch.abs(out*mask)).clone()
                out_prev = self.dropout_conv(out)
                # threshold_out.append(hoyer_thr)
                self.threshold_out.append(self.threshold['t'+str(l)].clone().detach())
                i += 1

        out_prev = out_prev.view(out_prev.size(0), -1)
        prev = len(self.features)

        for l in range(len(self.classifier)-1):
            # print('{} layer, length: {}'.format(l, len(self.classifier)))
            if isinstance(self.classifier[l], (nn.Linear)):
                out_prev = self.classifier[l](out_prev) #- getattr(self.threshold, 't'+str(prev+l))*epoch*1e-3
            
            if isinstance(self.classifier[l], (ThrBiAct, SubBiAct, HoyerBiAct)):
                # 1. relu:
                # out = self.relu(out_prev)
                # 2. x/thr -> act
                # out = out_prev
                out = out_prev/getattr(self.threshold, 't'+str(prev+l))
                # out = out_prev/torch.abs(getattr(self.threshold, 't'+str(prev+l)))
                self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), prev+l)
                out = self.classifier[l](out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], self.x_thr_scale, prev+l, self.start_spike_layer)

                # 3. x/hoyer_thr -> act
                # out = self.relu(out_prev)
                # self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), prev+l)
                # # hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
                # hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
                # out = out/hoyer_thr
                # out = self.act_func(out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], prev+l, self.start_spike_layer)

                # 4.
                # out = out_prev/torch.abs(getattr(self.threshold, 't'+str(prev+l)))
                # out = self.relu(out)
                # out = torch.clamp(out, max=1.0)
                # self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), prev+l)
                # out /= (torch.sum((out.clone())**2) / torch.sum(torch.abs(out.clone()))) # 1819 has 0.618
                # out = self.classifier[l](out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], prev+l, self.start_spike_layer)
                
                self.relu_batch_num += self.num_relu(out, 0.0, 1.0, torch.max(out).clone().detach())
                if epoch == -1:
                    act_out[prev+l] = out.clone().detach()
                # elif epoch == -2:
                #     act_out[prev+l] = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1)) # hoyer threshold
                else:
                    if torch.sum(torch.abs(out))>0: # and prev+l < self.start_spike_layer
                        if self.hoyer_type == 'mean':
                            act_out += (torch.mean(torch.sum(torch.abs(out), dim=1)**2 / torch.sum((out)**2, dim=1))).clone()
                        elif self.hoyer_type == 'sum':
                            act_out +=  (torch.sum(torch.abs(out))**2 / torch.sum((out)**2)).clone()
                        # elif self.hoyer_type == 'mask':
                        #     mask = torch.zeros_like(out).cuda()
                        #     mask[out<torch.max(out).clone().detach()] = 1.0
                        #     if torch.sum(torch.abs(out*mask))>0:
                        #         act_out += (torch.mean(torch.sum(torch.abs(out*mask), dim=1)**2 / torch.sum((out*mask)**2, dim=1))).clone()
                        # elif self.hoyer_type == 'l1':
                        #     mask = torch.zeros_like(out).cuda()
                        #     mask[out<torch.max(out).clone().detach()] = 1.0
                        #     act_out += torch.sum(torch.abs(out*mask)).clone()
                        
                out_prev = self.dropout_linear(out)
                # threshold_out.append(hoyer_thr)
                self.threshold_out.append(self.threshold['t'+str(prev+l)].clone().detach())
                i += 1
        # print(self.classifier[l+1])
        out = self.classifier[l+1](out_prev)

        # return out, act_out
        return out, self.threshold_out, self.relu_batch_num, act_out #self.layer_output
        #out = self.features(x)
        #out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        #return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    
    def _make_layers(self, cfg):
        act_size = 32
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        # index = 0
        for i,x in enumerate(cfg):
            stride = 1
            
            
            if x == 'A':
                pass
            else:
                if self.conv_type == 'ori':
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True)
                elif self.conv_type == 'dy':
                    conv2d = Dynamic_conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, bias=False)
                
                # elif self.conv_type == 'biconv':
                # if index >= 12:
                #     print(index)
                #     conv2d = BiConv(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True)
                if i+1 < len(cfg) and cfg[i+1] == 'A':
                    if self.pool_pos == 'before_relu':
                        act_size //= 2
                        layers += [conv2d,
                                # nn.AvgPool2d(kernel_size=2, stride=2),
                                self.pooling,
                                nn.BatchNorm2d(x) if self.bn_type == 'bn' else tdBatchNorm(x),
                                HoyerBiAct(num_features=x, hoyer_type=self.act_mode)
                                # SubBiAct(act_mode=self.act_mode, bit=1, act_size=(x, act_size, act_size)) if self.sub_act_mask else ThrBiAct(act_mode=self.act_mode) 
                                ]
                    elif self.pool_pos == 'after_relu':
                        layers += [conv2d,
                                nn.BatchNorm2d(x) if self.bn_type == 'bn' else tdBatchNorm(x),
                                HoyerBiAct(num_features=x, hoyer_type=self.act_mode),
                                # SubBiAct(act_mode=self.act_mode, bit=1, act_size=(x, act_size, act_size)) if self.sub_act_mask else ThrBiAct(act_mode=self.act_mode) ,
                                self.pooling,
                                # nn.AvgPool2d(kernel_size=2, stride=2)
                                ]
                        act_size //= 2
                else:
                    layers += [conv2d,
                            nn.BatchNorm2d(x) if self.bn_type == 'bn' else tdBatchNorm(x),
                            HoyerBiAct(num_features=x, hoyer_type=self.act_mode),
                            # SubBiAct(act_mode=self.act_mode, bit=1, act_size=(x, act_size, act_size)) if self.sub_act_mask else ThrBiAct(act_mode=self.act_mode)
                            ]
                #layers += [nn.Dropout(self.dropout)]           
                in_channels = x
        
        return nn.Sequential(*layers)

def test():
    net = VGG_TUNABLE_THRESHOLD_tdbn(act_mode='cw', conv_type='ori', start_spike_layer=0).cuda()
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(2,3,32,32).cuda()
    y,_,_,_ = net(x)
    target = torch.randn(y.shape).cuda()
    loss = F.cross_entropy(y,target)
    loss.backward()
    print(y[0].size())
    # For VGG6 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG6')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())

def test2():
    x = torch.randn(2,3,32,32)
    conv2d = Dynamic_conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    y = conv2d(x)
    print(x.shape, y.shape, isinstance(conv2d, Dynamic_conv2d))

if __name__ == '__main__':
    from dynamic_conv import Dynamic_conv2d
    test()
