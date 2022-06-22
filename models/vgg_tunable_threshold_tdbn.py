import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from models.dynamic_conv import Dynamic_conv2d
# from dynamic_conv import Dynamic_conv2d

class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接:https://arxiv.org/pdf/2011.05280。具体是在BN时,也在时间域上作平均;并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=0.25):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth

    def forward(self, input, Vth):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.alpha * Vth * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

cfg = {
    'VGG4' : [64, 'A', 128, 'A'],
    'VGG6' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512], # 13 conv + 3 fc
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}

class Threshold_relu(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = relu(input-min_thr_scale)
        # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
        out[input >= max_thr_scale] = 1.0

        # ctx.x_scale = 1.0
        # out[out >= ctx.x_scale*hoyer_thr] = 1.0
        # out[out < ctx.x_scale*hoyer_thr] = 0.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        # grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        # grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0
        grad_inp[input > (-1.0*0.0)] = 1.0
        grad_inp[input > (2.0*1.0)] = 0.0

        # print('layer: {}, grad_output_norm: {:.2f}, grad_inp_norm: {:.2f}, grad_out_norm: {:.2f}'.format(
        #     ctx.layer_index, torch.norm(grad_output, p=2), torch.norm(grad_inp, p=2), torch.norm(grad_inp*grad_input, p=2)))

        return grad_inp*grad_input, None, None, None, None, None
        # return grad_inp*grad_input, grad_threshold*grad_inp*grad_thr, None, None, None, None
class Threshold_mean(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        out = relu(input-min_thr_scale)
        out[input >= max_thr_scale] = 1.0
        # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
        if layer_index >= 44:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
        else:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
        ctx.x_scale = 1.0
        # ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        if layer_index >= start_spike_layer:
            out[out <= ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        # grad_inp[input > (2.0*ctx.x_scale*ctx.hoyer_thr)] = 0.0
        grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0

        return grad_inp*grad_input, None, None, None, None, None
class Threshold_sum(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        out = relu(input-min_thr_scale)
        out[input >= max_thr_scale] = 1.0

        hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
        
        ctx.x_scale = 1.0
        # ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        if layer_index >= start_spike_layer:
            out[out <= ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        # grad_inp[input > (2.0*ctx.x_scale*ctx.hoyer_thr)] = 0.0
        grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0

        return grad_inp*grad_input, None, None, None, None, None

class Threshold_relu_v2(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, threshold=1.0):
        
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = input/threshold
        out = relu(input-min_thr_scale)
        # 0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18
        out[input >= max_thr_scale] = 1.0
        # 0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18
        out = out*float(threshold)
        ctx.save_for_backward(input, threshold, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,threshold,output,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input, grad_threshold = grad_output.clone(), grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)*threshold] = 1.0
        grad_inp[input > (2.0*ctx.max_thr_scale)*threshold] = 0.0

        grad_thr = (-1*threshold*output - input) / threshold**2

        return grad_inp*grad_input/threshold, None, None, None, None, grad_inp*grad_threshold*grad_thr

class Threshold_relu_v3(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale, ctx.layer_index = min_thr_scale, max_thr_scale, layer_index
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = relu(input-min_thr_scale)
        # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
        out[input >= max_thr_scale] = 1.0 
        if layer_index >= 44:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
        else:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
        ctx.x_scale = 1.0
        if layer_index >= start_spike_layer:
            # print('1. layer: {}, hoyer_thr: {}'.format(layer_index, hoyer_thr))
            out[out < ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        # ctx.hoyer_thr = hoyer_thr
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0
        # grad_inp[input > (2.0*ctx.hoyer_thr)] = 0.0

        # HOYER-SQUARE REGULARIZER part  mask vs no mask
        # mask = out<threshold
        # abs_sum = torch.sum(torch.abs(out))
        # sq_sum = torch.sum((out)**2)
        # hs_grad = 2*abs_sum/(sq_sum**2)*torch.sign(out)*(sq_sum - abs_sum*torch.abs(out))

        # print('layer: {}, grad_output: {}, sum: {:.2f}, hs_grad: {}, sum: {:.2f}, grad_inp: {}, out: {:.2f}'.format(
        #     ctx.layer_index, grad_output.shape, torch.norm(grad_output, p=2), \
        #         hs_grad.shape, torch.norm(hs_grad,p=2), \
        #         torch.norm(grad_inp), torch.norm(grad_inp*(grad_input + 1e-6*hs_grad))))
        return grad_inp*grad_input, None, None, None, None, None
        # return grad_inp*(grad_input + 1e-8*hs_grad), None, None, None, None
class Threshold_relu_v4(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0):
        
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = relu(input-min_thr_scale)
        out[out >=1.0] = 1.0
       
        # if layer_index >= 44:
        #     hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
        # else:
        #     hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
        # out[out>=hoyer_thr] = 1.0
        # out[out<hoyer_thr] = 0.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0


        return grad_inp*grad_input, None, None, None, None

class Threshold_cw(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = relu(input-min_thr_scale)
        # 0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18
        out[input >= max_thr_scale] = 1.0
        if layer_index >= 44:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
        else:
            hoyer_cw = torch.sum((out)**2, dim=(2,3)) / torch.sum(torch.abs(out), dim=(2,3))
            hoyer_cw = torch.nan_to_num(hoyer_cw, nan=0.0)
            hoyer_cw = torch.mean(hoyer_cw, dim=0)
            N,C,W,H = input.shape
            hoyer_thr = torch.permute(hoyer_cw*(torch.ones(N,W,H,C)).cuda(), (0,3,1,2))
            # print('mean: {}, sum: {}, cw: {}'.format(hoyer_mean, torch.mean(hoyer_sum), torch.mean(hoyer_thr)))
            # print('layer: {}, hoyer_thr: {}'.format(layer_index, torch.mean(hoyer_thr)))
        ctx.x_scale = 1.0
        # ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        if layer_index >= start_spike_layer:
            # print('1. layer: {}, hoyer_thr: {}'.format(layer_index, hoyer_thr))
            out[out <= ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        
        # print('2. layer: {}, hoyer_cw: {}'.format(layer_index, hoyer_cw))
        # exit()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
        # grad_inp[input > (2.0*ctx.x_scale*ctx.hoyer_thr)] = 0.0
        grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0

        # print('layer: {}, grad_output_norm: {:.2f}, grad_inp_norm: {:.2f}, grad_out_norm: {:.2f}'.format(
        #     ctx.layer_index, torch.norm(grad_output, p=2), torch.norm(grad_inp, p=2), torch.norm(grad_inp*grad_input, p=2)))

        return grad_inp*grad_input, None, None, None, None, None

class VGG_TUNABLE_THRESHOLD_tdbn(nn.Module):
    def __init__(self, vgg_name='VGG16', labels=10, dataset = 'CIFAR10', kernel_size=3, linear_dropout=0.1, conv_dropout=0.1, default_threshold=1.0, \
        net_mode='ori', hoyer_type='mean', act_mode = 'mean', bn_type='bn', start_spike_layer=50, conv_type='ori'):
        super(VGG_TUNABLE_THRESHOLD_tdbn, self).__init__()
        
        self.dataset        = dataset
        self.kernel_size    = kernel_size
        self.bn_type        = bn_type
        self.conv_type      = conv_type
        self.features       = self._make_layers(cfg[vgg_name])
        self.dropout_conv   = nn.Dropout(conv_dropout)
        self.dropout_linear = nn.Dropout(linear_dropout)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()
        self.hoyer_type     = hoyer_type
        self.start_spike_layer = start_spike_layer
        self.test_hoyer_thr = torch.tensor([0.0]*15)

        if act_mode == 'fixed':
            self.act_func = Threshold_relu.apply
        elif act_mode == 'mean':
            self.act_func = Threshold_mean.apply
        elif act_mode == 'sum':
            self.act_func = Threshold_sum.apply
        elif act_mode == 'cw':
            self.act_func = Threshold_cw.apply
    
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
                            nn.ReLU(inplace=True),
                            #nn.Dropout(0.5),
                            nn.Linear(4096, 4096, bias=True),
                            nn.ReLU(inplace=True),
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
            if isinstance(self.features[l], nn.ReLU):
                self.threshold['t'+str(l)] 	= nn.Parameter(torch.tensor(default_threshold))
                #percentile['t'+str(l)]  = nn.Parameter(torch.ones(9))
        prev = len(self.features)
        for l in range(len(self.classifier)-1):#-1
            if isinstance(self.classifier[l], nn.ReLU):
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
        
        out = input
        out[out < 0.0] = 0.0
        out[out >= 1.0] = 1.0
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
        threshold_out = []
        relu_total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        if epoch < 0:
            act_out = {}
        else:
            act_out = 0.0
        i = 0
        for l in range(len(self.features)):
            if isinstance(self.features[l], (nn.Conv2d, nn.MaxPool2d, Dynamic_conv2d)):
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
            if isinstance(self.features[l], (nn.ReLU)):
                # print('{}, shape: {}'.format(l, out_prev.shape))
                # relu:
                # out = self.relu(out_prev)
                
                out = out_prev/getattr(self.threshold, 't'+str(l))
                self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), l)
                out = self.act_func(out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], l, self.start_spike_layer)
                
                relu_total_num += self.num_relu(out.clone().detach(), 0.0, 1.0, torch.max(out).clone().detach())
                if epoch == -1:
                    act_out[l] = out.clone().detach()
                elif epoch == -2:
                    act_out[l] = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3))) # hoyer threshold
                else:
                    if torch.sum(torch.abs(out))>0 and l < self.start_spike_layer:
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
                threshold_out.append(self.threshold['t'+str(l)])
                i += 1

        out_prev = out_prev.view(out_prev.size(0), -1)
        prev = len(self.features)

        for l in range(len(self.classifier)-1):
            # print('{} layer, length: {}'.format(l, len(self.classifier)))
            if isinstance(self.classifier[l], (nn.Linear)):
                out_prev = self.classifier[l](out_prev) #- getattr(self.threshold, 't'+str(prev+l))*epoch*1e-3
            
            if isinstance(self.classifier[l], (nn.ReLU)):
                # relu
                # out = self.relu(out_prev)
                
                out = out_prev/getattr(self.threshold, 't'+str(prev+l))
                self.test_hoyer_thr[i] = self.get_hoyer_thr(out.clone().detach(), prev+l)
                out = self.act_func(out, epoch, self.min_thr_scale[i], self.max_thr_scale[i], prev+l, self.start_spike_layer)
                
                relu_total_num += self.num_relu(out, 0.0, 1.0, torch.max(out).clone().detach())
                if epoch == -1:
                    act_out[prev+l] = out.clone().detach()
                elif epoch == -2:
                    act_out[prev+l] = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1)) # hoyer threshold
                else:
                    if torch.sum(torch.abs(out))>0 and prev+l < self.start_spike_layer:
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
                threshold_out.append(self.threshold['t'+str(prev+l)])
                i += 1
        # print(self.classifier[l+1])
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
                if self.conv_type == 'ori':
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=True)
                elif self.conv_type == 'dy':
                    conv2d = Dynamic_conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, bias=False)
                layers += [conv2d,
                        nn.BatchNorm2d(x) if self.bn_type == 'bn' else tdBatchNorm(x),
                        # nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                           #nn.ReLU(inplace=True)
                           #]
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
