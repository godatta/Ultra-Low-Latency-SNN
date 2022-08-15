import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.dynamic_conv import Dynamic_conv2d
from models.self_modules import HoyerBiAct

cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512], # 13 conv + 3 fc
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


class VGG_TUNABLE_THRESHOLD_tdbn_imagenet(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', hoyer_type='mean', act_mode = 'mean', bn_type='bn', start_spike_layer=50, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=1, im_size=224):
        super(VGG_TUNABLE_THRESHOLD_tdbn_imagenet, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.bn_type        = bn_type
        self.conv_type      = conv_type
        self.pool_pos       = pool_pos
        self.act_mode       = act_mode
        self.sub_act_mask   = sub_act_mask
        self.hoyer_type     = hoyer_type
        self.if_spike       = True if start_spike_layer == 0 else False 
        self.x_thr_scale    = x_thr_scale
        self.weight_quantize= weight_quantize
        self.pooling        = nn.MaxPool2d(kernel_size=2, stride=2) if pooling_type == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)
        self.features       = self._make_layers(cfg[vgg_name])
        self.dropout_conv   = nn.Dropout(conv_dropout)
        self.dropout_linear = nn.Dropout(linear_dropout)

        self.threshold      = {}
            

        if net_mode == 'ori':
            self.min_thr_scale = [0.0]*15
            self.max_thr_scale = [1.0]*15
        
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
        elif vgg_name!='VGG6' and dataset=='IMAGENET':
            self.classifier = nn.Sequential(
                            nn.Linear((im_size//32)**2*512, 4096, bias=False),
                            # nn.Linear(512*7*7, 4096, bias=False),
                            HoyerBiAct(hoyer_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            nn.Linear(4096, 4096, bias=False),
                            HoyerBiAct(hoyer_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            nn.Linear(4096, labels, bias=False)
            )
        elif vgg_name!='VGG6' and dataset!='MNIST':
            self.classifier = nn.Sequential(
                            nn.Linear(2048, 4096, bias=False),
                            HoyerBiAct(hoyer_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=False),
                            HoyerBiAct(hoyer_type='sum', x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, labels, bias=False)
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
        self._initialize_weights2()
    
    def forward(self, x):   #######epoch
        out_prev = x
        act_out = 0.0
        for l in range(len(self.features)):
            if isinstance(self.features[l], HoyerBiAct):
                
                out_prev = self.features[l](out_prev)
                if torch.sum(torch.abs(out_prev))>0: #  and l < self.start_spike_layer
                    # out_copy = out_prev.clone()
                    # act_out +=  (torch.sum(torch.abs(out_copy))**2 / torch.sum((out_copy)**2))
                    act_out +=  (torch.sum(torch.abs(out_prev))**2 / torch.sum((out_prev)**2)).clone()
                out_prev = self.dropout_conv(out_prev)
            else:
                out_prev = self.features[l](out_prev)
   
        out_prev = out_prev.view(out_prev.size(0), -1)

        for l in range(len(self.classifier)-1):
            if isinstance(self.classifier[l], nn.Linear):
                out_prev = self.classifier[l](out_prev) #- getattr(self.threshold, 't'+str(prev+l))*epoch*1e-3
            
            if isinstance(self.classifier[l], HoyerBiAct):
                
                out_prev = self.classifier[l](out_prev)  
                if torch.sum(torch.abs(out_prev))>0: # and prev+l < self.start_spike_layer
                    # out_copy = out_prev.clone()
                    # act_out +=  (torch.sum(torch.abs(out_copy))**2 / torch.sum((out_copy)**2))    
                    act_out +=  (torch.sum(torch.abs(out_prev))**2 / torch.sum((out_prev)**2)).clone() 
                out_prev = self.dropout_linear(out_prev)

        out = self.classifier[l+1](out_prev)

        return out, act_out #self.layer_output


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
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False)
                elif self.conv_type == 'dy':
                    conv2d = Dynamic_conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, bias=False)

                if i+1 < len(cfg) and cfg[i+1] == 'A':
                    if self.pool_pos == 'before_relu':
                        act_size //= 2
                        layers += [conv2d,
                                self.pooling,
                                nn.BatchNorm2d(x),
                                HoyerBiAct(num_features=x, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike)
                                ]
                    elif self.pool_pos == 'after_relu':
                        layers += [conv2d,
                                nn.BatchNorm2d(x),
                                HoyerBiAct(num_features=x, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                                self.pooling,
                                ]
                        act_size //= 2
                else:
                    layers += [conv2d,
                            nn.BatchNorm2d(x),
                            HoyerBiAct(num_features=x, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike),
                            ]
                in_channels = x
        if self.dataset == 'IMAGENET':
            layers.pop()
            layers.pop()

            layers += [self.pooling, 
                    nn.BatchNorm2d(x),
                    HoyerBiAct(num_features=x, hoyer_type=self.act_mode, x_thr_scale=self.x_thr_scale, if_spike=self.if_spike)]
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