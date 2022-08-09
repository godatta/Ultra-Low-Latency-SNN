from models.self_modules import HoyerBiAct
import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, conv_dropout, threshold1, threshold2, hoyer_type):
        super().__init__()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.spike1 = HoyerBiAct(in_planes, hoyer_type=hoyer_type)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike2 = HoyerBiAct(planes, hoyer_type=hoyer_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            raise RuntimeError('not considered situation')
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                
            )

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    # spike -> conv -> bn -> spike -> conv -> bn -> + x
    def forward(self, x):
        out = x/self.threshold1
        out = self.spike1(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv_dropout(out)

        out = out/self.threshold2
        out = self.spike2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.identity(x)

        return out

class ResNet_hoyer(nn.Module):
    def __init__(self, block, num_blocks, labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', hoyer_type='mean', act_mode = 'mean', bn_type='bn', start_spike_layer=50, conv_type='ori', pool_pos='after_relu', sub_act_mask=False, \
        x_thr_scale=1.0, pooling_type='max', weight_quantize=1, im_size=224):
        
        super(ResNet_hoyer, self).__init__()

        self.in_planes      = 64
        self.conv_dropout   = conv_dropout
        self.thr_index      = 0
        threshold           = {}
        self.perc = {}
        self.pre_process    = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                HoyerBiAct(num_features=64, hoyer_type=self.act_mode),
                                nn.Dropout(self.conv_dropout),

                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                HoyerBiAct(num_features=64, hoyer_type=self.act_mode),
                                nn.Dropout(self.conv_dropout),

                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2),
                                nn.BatchNorm2d(64),
                                # HoyerBiAct(num_features=64, hoyer_type=self.act_mode)
                                )
        
        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], HoyerBiAct):
				#self.register_buffer('threshold[l]', torch.tensor(default_threshold, requires_grad=True))
                threshold['t'+str(self.thr_index)] 	= nn.Parameter(torch.tensor(default_threshold))
                self.thr_index += 1
       
        self.threshold 	= nn.ParameterDict(threshold)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, conv_dropout=self.conv_dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, conv_dropout=self.conv_dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, conv_dropout=self.conv_dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, conv_dropout=self.conv_dropout)
        self.classifier     = nn.Sequential(
                                nn.Linear(2048, labels, bias=False)
                                )
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                l += 1
                if isinstance(layer, HoyerBiAct):
                    threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))

        #self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4:self.layer4}
        
        
        self._initialize_weights2()

    def _initialize_weights2(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def _make_layer(self, block, planes, num_blocks, stride, conv_dropout, hoyer_type):
        
        if num_blocks==0:
            return nn.Sequential()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            threshold1 = getattr(self.threshold, 't'+str(self.thr_index))
            threshold2 = getattr(self.threshold, 't'+str(self.thr_index+1))
            self.thr_index += 2
            layers.append(block(self.in_planes, planes, stride, conv_dropout, threshold1, threshold2, hoyer_type))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def percentile(self, t, q):

        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, x):

        out_prev = x
        threshold_out = []
        j = 0

        for l in range(len(self.pre_process)):
            if isinstance(self.pre_process[l], nn.Conv2d):
                #self.perc[str(j)] = {}
                #for i in range(1, 100):
                #    self.perc[str(j)][str(i)] = self.percentile(self.pre_process[l](out_prev).view(-1), i)
                #j += 1
                out = self.pre_process[l](out_prev)
                out[out<0] = 0
                out[out>getattr(self.threshold, 't'+str(l))] =  getattr(self.threshold, 't'+str(l))
                out_prev = out.clone()
                threshold_out.append(getattr(self.threshold, 't'+str(l)))
            else:
                out_prev = self.pre_process[l](out_prev)
        pos = len(self.pre_process)
        out = out_prev
        j = 0


        for layer in self.layer1:
            out, a, b = layer(out)

        for layer in self.layer2:
            out, a, b = layer(out)

        for layer in self.layer3:
            out, a, b = layer(out)

        for layer in self.layer4:
            out, a, b = layer(out)

                
        #out = self.layer1(out_prev)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(o    ut)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        for l in range(pos,self.pos4):
            threshold_out.append(getattr(self.threshold, 't'+str(l)))
        return out, threshold_out
        
def ResNet12(labels=10, conv_dropout=0.2, default_threshold=1.0):
    return ResNet_hoyer(block=BasicBlock, num_blocks=[1,1,1,1], labels=labels, conv_dropout=conv_dropout, default_threshold=default_threshold)

def ResNet20(labels=10, conv_dropout=0.2, default_threshold=1.0):
    return ResNet_hoyer(block=BasicBlock, num_blocks=[2,2,2,2], labels=labels, conv_dropout=conv_dropout, default_threshold=default_threshold)

def ResNet34(labels=10, conv_dropout=0.2, default_threshold=1.0):
    return ResNet_hoyer(block=BasicBlock, num_blocks=[3,4,5,3], labels=labels, conv_dropout=conv_dropout, default_threshold=default_threshold)

def test():
    print('In test()')
    net = ResNet12()
    print('Calling y=net() from test()')
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == '__main__':
    test()
