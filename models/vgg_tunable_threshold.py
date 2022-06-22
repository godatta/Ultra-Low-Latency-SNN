import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math



cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512], # 13 conv + 3 fc
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}


# class Threshold_relu(torch.autograd.Function):
#     """
#     Here we use the piecewise-linear surrogate gradient as was done
#     in Bellec et al. (2018).
#     """
#     #gamma = 0.3 #Controls the dampening of the piecewise-linear surrogate gradient

#     @staticmethod
#     def forward(ctx, input, threshold, epoch, min_thr_scale=0.0, max_thr_scale=0.0):
        
#         ctx.save_for_backward(input, threshold)
#         ctx.epoch = epoch
#         ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
#         relu = nn.ReLU()

#         # out = relu(input)
#         out = relu(input-min_thr_scale*threshold)
#         out[out >= (max_thr_scale*threshold)] = threshold
#         # out = relu(input-threshold*epoch*1e-3)*(threshold)/(threshold - threshold*epoch*2e-3)
#         # out[out > (threshold - threshold*epoch*1e-3)] = threshold
#         # out = torch.zeros_like(input).cuda()
#         # out[out > (max_thr_scale*threshold)] = threshold
        
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
        
#         input, threshold    = ctx.saved_tensors
#         epoch = ctx.epoch
#         grad_input, grad_threshold = grad_output.clone(), grad_output.clone()
#         grad_inp, grad_thr = torch.zeros_like(input).cuda(), torch.zeros_like(input).cuda()
#         # grad_inp[input > -1.0*ctx.min_thr_scale*threshold] = 1.0
#         grad_inp[input < 0] = 1.0
#         grad_inp[input > threshold] = 0.0
#         grad_thr[input > 0] = 1.0

#         # grad_inp[input > threshold*(ctx.min_thr_scale)] = 1.0
#         # grad_inp[input > threshold*(-1.0*ctx.min_thr_scale)] = 1.0
#         # grad_inp[input > threshold*(2.0*ctx.max_thr_scale)] = 0.0
#         # grad_thr[input > threshold*ctx.min_thr_scale] = 1.0	

#         # grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
#         # grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0
        	

#         #grad[input <=- 1] = 0.0
#         #grad = input
#         #grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
#         return grad_inp*grad_input, grad_thr*grad_threshold, None, None, None
#         #return last_spike*grad_input, None


class Threshold_relu(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    #gamma = 0.3 #Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, threshold, epoch, min_thr_scale=0.0, max_thr_scale=1.0):
        ctx.save_for_backward(input, threshold)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        relu = nn.ReLU()

        out = torch.zeros_like(input).cuda()
        out = relu(input-threshold*(min_thr_scale ))
        out[out >= (threshold*(max_thr_scale))] = float(threshold)
        # output_thr = torch.mean(out[out>=(threshold*(max_thr_scale - epoch*1e-3))])
        # out[out >= (threshold*(max_thr_scale - epoch*1e-3))] = float(output_thr)

        # out = relu(input-threshold*(min_thr_scale + (epoch//20)*1e-3))
        # output_thr = torch.mean(out[out>=(threshold*(max_thr_scale))])
        # out[out >= (threshold*(max_thr_scale))] = float(output_thr)


        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input, threshold  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input, grad_threshold = grad_output.clone(), grad_output.clone()
        grad_inp, grad_thr = torch.zeros_like(input).cuda(), torch.zeros_like(input).cuda()

        # v0
        # grad_inp[input > 0.0] = 1.0
        # grad_inp[input > threshold] = 0.0
        # grad_thr[input > threshold*ctx.min_thr_scale] = 1.0	
        # v1
        grad_inp[input > threshold*(-1.0*ctx.min_thr_scale)] = 1.0
        grad_inp[input > threshold*(2.0*ctx.max_thr_scale)] = 0.0
        grad_thr[input > threshold*ctx.min_thr_scale] = 1.0	

        # HOYER-SQUARE REGULARIZER part  mask vs no mask
        # mask = out<threshold
        # abs_sum = torch.sum(torch.abs(out))
        # sq_sum = torch.sum((out)**2)
        # hs_grad = 2*abs_sum/(sq_sum**2)*torch.sign(out)*(sq_sum - abs_sum*torch.abs(out))
        # return grad_inp*(grad_input + 1e-8*hs_grad), grad_thr*grad_input, None, None, None
        # return grad_inp*grad_input, grad_thr*grad_threshold, None, None, None
        return grad_inp*grad_input, grad_thr*grad_threshold, None, None, None


class VGG_TUNABLE_THRESHOLD(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1, mode='cut'):
        super(VGG_TUNABLE_THRESHOLD, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.features       = self._make_layers(cfg[vgg_name])
        self.dropout_conv   = nn.Dropout(conv_dropout)
        self.dropout_linear = nn.Dropout(linear_dropout)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()
        self.act_func 		= Threshold_relu.apply
        self.threshold 	= {}
        self.perc = {}
        self.dis = []
        self.layer_output = {}
        with open('output/ann_min_scale_vgg16_cifar10_0.3', 'rb') as f:
            self.min_thr_scale = torch.load(f)
        with open('output/ann_max_scale_vgg16_cifar10_0.7', 'rb') as f:
            self.max_thr_scale = torch.load(f) 
        # with open('output/ann_min_scale_vgg16_cifar10_0.2', 'rb') as f:
        #     self.min_thr_scale = torch.load(f)
        # with open('output/ann_max_scale_vgg16_cifar10_0.8', 'rb') as f:
        #     self.max_thr_scale = torch.load(f)

        if mode == 'ori':
            self.min_thr_scale = [0.0]*len(self.min_thr_scale)
            self.max_thr_scale = [1.0]*len(self.max_thr_scale)
        
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
                            #nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=True),
                            #nn.ReLU(inplace=True),
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
            if isinstance(self.features[l], nn.Conv2d):
                self.threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(l)]  = nn.Parameter(torch.ones(9))
        prev = len(self.features)
        for l in range(len(self.classifier)-1):#-1
            if isinstance(self.classifier[l], nn.Linear):
                self.threshold['t'+str(prev+l)]	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(prev+l)]  = nn.Parameter(torch.ones(9))
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

    def forward(self, x, epoch=1):   #######epoch
        out_prev = x
        threshold_out = []
        relu_total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if epoch == -1:
            act_out = {}
        else:
            act_out = 0.0
        i = 0
        for l in range(len(self.features)):
            if isinstance(self.features[l], (nn.Conv2d)):

                out = self.features[l](out_prev) 
                # hoyer_thr = torch.mean(torch.sum(out**2, dim=(1,2,3))) / torch.mean(torch.sum(torch.abs(out), dim=(1,2,3)))
                # relu_total_num += self.num_relu(out, self.min_thr_scale[i], self.max_thr_scale[i], hoyer_thr)
                # out = self.act_func(out, hoyer_thr, epoch, self.min_thr_scale[i], self.max_thr_scale[i])
                relu_total_num += self.num_relu(out, self.min_thr_scale[i], self.max_thr_scale[i], getattr(self.threshold, 't'+str(l)))
                # out = self.relu(out)
                out = self.act_func(out, getattr(self.threshold, 't'+str(l)), epoch, self.min_thr_scale[i], self.max_thr_scale[i])
                if epoch == -1:
                    act_out[l] = out.clone()
                else:
                    if torch.sum(torch.abs(out))>0:
                        # mask = torch.zeros_like(out).cuda()
                        # mask[out<getattr(self.threshold, 't'+str(l))] = 1.0
                        # act_out +=  (torch.sum(torch.abs(out*mask))**2 / torch.sum((out*mask)**2)).clone() 
                        # act_out +=  (torch.sum(torch.abs(out))**2 / torch.sum((out)**2)).clone()
                        act_out += (torch.mean(torch.sum(torch.abs(out), dim=(1,2,3))**2) / torch.mean(torch.sum(out**2, dim=(1,2,3)))).clone()
                        # act_out += torch.sum(out)/out.shape[0]

                out_prev = self.dropout_conv(out)
                # threshold_out.append(hoyer_thr)
                threshold_out.append(self.threshold['t'+str(l)])
                i += 1
                #threshold_out.append(getattr(self.threshold, 't'+str(l)))
            if isinstance(self.features[l], (nn.MaxPool2d)):
                out_prev = self.features[l](out_prev)

        out_prev = out_prev.view(out_prev.size(0), -1)
        prev = len(self.features)

        for l in range(len(self.classifier)-1):
            if isinstance(self.classifier[l], (nn.Linear)):

                out = self.classifier[l](out_prev) #- getattr(self.threshold, 't'+str(prev+l))*epoch*1e-3
                # hoyer_thr = torch.mean(torch.sum(out**2, dim=1)) / torch.mean(torch.sum(torch.abs(out), dim=1)).clone().detach()
                # relu_total_num += self.num_relu(out, self.min_thr_scale[i], self.max_thr_scale[i], hoyer_thr)
                # out = self.act_func(out, hoyer_thr, epoch, self.min_thr_scale[i], self.max_thr_scale[i])
                relu_total_num += self.num_relu(out, self.min_thr_scale[i], self.max_thr_scale[i], getattr(self.threshold, 't'+str(prev+l)))
                # out = self.relu(out)
                out = self.act_func(out, getattr(self.threshold, 't'+str(prev+l)), epoch, self.min_thr_scale[i], self.max_thr_scale[i])
                if epoch == -1:
                    act_out[prev+l] = out.clone()
                else:
                    if torch.sum(torch.abs(out))>0:
                        # mask = torch.zeros_like(out).cuda()
                        # mask[out<getattr(self.threshold, 't'+str(prev+l))] = 1.0
                        # act_out +=  (torch.sum(torch.abs(out*mask))**2 / torch.sum((out*mask)**2)).clone()
                        # act_out +=  (torch.sum(torch.abs(out))**2 / torch.sum((out)**2)).clone()
                        act_out += (torch.mean(torch.sum(torch.abs(out), dim=1)**2) / torch.mean(torch.sum(out**2, dim=1))).clone()
                        # act_out += torch.sum(out)/out.shape[0]
                out_prev = self.dropout_linear(out)
                # threshold_out.append(hoyer_thr)
                threshold_out.append(self.threshold['t'+str(prev+l)])
                i += 1

            out = self.classifier[l+1](out_prev)
        return out, threshold_out, relu_total_num, act_out #self.layer_output

    


        

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
        layers = []

        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3
        
        for x in cfg:
            stride = 1
            
            if x == 'A':
                #layers.pop()
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True)]
                           #nn.ReLU(inplace=True)
                           #]
                #layers += [nn.Dropout(self.dropout)]           
                in_channels = x

        
        return nn.Sequential(*layers)

def test():
    for a in cfg.keys():
        if a=='VGG6' or a=='VGG4':
            continue
        net = VGG(a)
        x = torch.randn(2,3,32,32)
        y = net(x)
        print(y.size())
    # For VGG6 change the linear layer in self. classifier from '512*2*2' to '512*4*4'    
    # net = VGG('VGG6')
    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
def test2():
    model = VGG_TUNABLE_THRESHOLD()
    state = torch.load('trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth', map_location='cpu')

    missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
    print('\n The threshold in snn is: {}'.format([model.threshold[key].data for key in model.threshold]))
    x = torch.randn(2,3,32,32)
    y = model(x)
    print(y.size())
if __name__ == '__main__':
    test2()
