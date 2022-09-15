from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# try combined threshold in hoyerBiAct
class HoyerBiAct(nn.Module):
    """
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    _version = 2
    __constants__ = ["num_features", "eps", "momentum", "spike_type", "x_thr_scale", "if_spike", "track_running_stats"]
    num_features: int
    eps: float
    momentum: float
    spike_type: str
    x_thr_scale: float
    if_spike: bool
    track_running_stats: bool
    # spike_type is args.act_mode
    def __init__(self, num_features=1, eps=1e-05, momentum=0.1, spike_type='sum', track_running_stats: bool = True, device=None, dtype=None, \
        min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=True):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HoyerBiAct, self).__init__()
        self.num_features   = num_features if spike_type == 'cw' else 1
        self.eps            = eps
        self.momentum       = momentum
        self.spike_type     = spike_type
        self.track_running_stats = track_running_stats
        self.threshold      = nn.Parameter(torch.tensor(1.0))
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = if_spike  
        # self.register_buffer('x_thr_scale', torch.tensor(x_thr_scale))
        # self.register_buffer('if_spike', torch.tensor(if_spike))
             

        # self.running_hoyer_thr = 0.0 if spike_type != 'cw' else torch.zeros(num_features).cuda()
        if self.track_running_stats:
            self.register_buffer('running_hoyer_thr', torch.zeros(self.num_features, **factory_kwargs))
            self.running_hoyer_thr: Optional[torch.Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_hoyer_thr", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_running_stats()
    
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_hoyer_thr/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_hoyer_thr.zero_()  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def forward(self, input):
        # calculate running estimates
        input = input / torch.abs(self.threshold)
        # input = torch.clamp(input, min=0.0, max=1.0)
        if self.training:
            clamped_input = torch.clamp((input).clone().detach(), min=0.0, max=1.0)
            # clamped_input = input.clone().detach()
            if self.spike_type == 'sum':
                hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
                # if torch.sum(torch.abs(clamped_input)) > 0:
                #     hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
                # else:
                #     print('Warning: the output is all zero!!!')

                #     hoyer_thr = self.running_hoyer_thr
            elif self.spike_type == 'fixed':
                hoyer_thr = 1.0                
            elif self.spike_type == 'cw':
                hoyer_thr = torch.sum((clamped_input)**2, dim=(0,2,3)) / torch.sum(torch.abs(clamped_input), dim=(0,2,3))
                # 1.0 is the max thr
                hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
                # hoyer_thr = torch.mean(hoyer_cw, dim=0)
            
            with torch.no_grad():
                self.running_hoyer_thr = self.momentum * hoyer_thr\
                    + (1 - self.momentum) * self.running_hoyer_thr
        else:
            hoyer_thr = self.running_hoyer_thr
            # only for test
            # if self.num_features == -1 or self.spike_type == 'sum':
            #     hoyer_thr =torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
            # if self.spike_type == 'fixed':
            #     hoyer_thr = 1.0                
            # elif self.spike_type == 'cw':
            #     hoyer_thr =torch.sum((clamped_input)**2, dim=(0,2,3)) / torch.sum(torch.abs(clamped_input), dim=(0,2,3))
            # print('running_hoyer_thr: {}'.format(self.running_hoyer_thr))
            
        # 
        out = Spike_func.apply(input, hoyer_thr, self.x_thr_scale, self.spike_type, self.if_spike)
        # input = Spike_func_with_thr.apply(input, hoyer_thr, self.x_thr_scale, self.spike_type, self.if_spike, self.threshold)
        # input = Spike_func.apply(input, hoyer_thr, x_thr_scale, self.spike_type, (layer_index>=13 and layer_index<=39))
        return out

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, spike_type={spike_type}, x_thr_scale={x_thr_scale}, if_spike={if_spike}, track_running_stats={track_running_stats}".format(**self.__dict__)
        )
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(HoyerBiAct, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
# ori one
# class HoyerBiAct(nn.Module):
#     """
#     Args:
#         num_features (int): same with nn.BatchNorm2d
#         eps (float): same with nn.BatchNorm2d
#         momentum (float): same with nn.BatchNorm2d
#         alpha (float): an addtional parameter which may change in resblock.
#         affine (bool): same with nn.BatchNorm2d
#         track_running_stats (bool): same with nn.BatchNorm2d
#     """
#     _version = 2
#     __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "spike_type"]
#     num_features: int
#     eps: float
#     momentum: float
#     spike_type: str
#     track_running_stats: bool
#     # spike_type is args.act_mode
#     def __init__(self, num_features=1, eps=1e-05, momentum=0.1, spike_type='sum',track_running_stats: bool = True, device=None, dtype=None):
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(HoyerBiAct, self).__init__()
#         self.num_features   = num_features if spike_type == 'cw' else 1
#         self.eps            = eps
#         self.momentum       = momentum
#         self.spike_type     = spike_type
#         self.track_running_stats = track_running_stats
#         # self.running_hoyer_thr = 0.0 if spike_type != 'cw' else torch.zeros(num_features).cuda()
#         if self.track_running_stats:
#             self.register_buffer('running_hoyer_thr', torch.zeros(self.num_features, **factory_kwargs))
#             self.running_hoyer_thr: Optional[torch.Tensor]
#             self.register_buffer('num_batches_tracked',
#                                  torch.tensor(0, dtype=torch.long,
#                                               **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
#         else:
#             self.register_buffer("running_hoyer_thr", None)
#             self.register_buffer("num_batches_tracked", None)
#         self.reset_running_stats()
    
#     def reset_running_stats(self) -> None:
#         if self.track_running_stats:
#             # running_hoyer_thr/num_batches... are registered at runtime depending
#             # if self.track_running_stats is on
#             self.running_hoyer_thr.zero_()  # type: ignore[union-attr]
#             self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

#     def forward(self, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
#         # calculate running estimates
#         if self.training:
#             clamped_input = torch.clamp(input.clone().detach(), min=0.0, max=1.0)
#             if self.spike_type == 'sum':
#                 hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
#                 # if torch.sum(torch.abs(clamped_input)) > 0:
#                 #     hoyer_thr = torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
#                 # else:
#                 #     print('Warning: the output is all zero!!!')

#                 #     hoyer_thr = self.running_hoyer_thr
#             elif self.spike_type == 'fixed':
#                 hoyer_thr = 1.0                
#             elif self.spike_type == 'cw':
#                 hoyer_thr = torch.sum((clamped_input)**2, dim=(0,2,3)) / torch.sum(torch.abs(clamped_input), dim=(0,2,3))
#                 # 1.0 is the max thr
#                 hoyer_thr = torch.nan_to_num(hoyer_thr, nan=1.0)
#                 # hoyer_thr = torch.mean(hoyer_cw, dim=0)
            
#             with torch.no_grad():
#                 self.running_hoyer_thr = self.momentum * hoyer_thr\
#                     + (1 - self.momentum) * self.running_hoyer_thr
#         else:
#             hoyer_thr = self.running_hoyer_thr
#             # only for test
#             # if self.num_features == -1 or self.spike_type == 'sum':
#             #     hoyer_thr =torch.sum((clamped_input)**2) / torch.sum(torch.abs(clamped_input))
#             # if self.spike_type == 'fixed':
#             #     hoyer_thr = 1.0                
#             # elif self.spike_type == 'cw':
#             #     hoyer_thr =torch.sum((clamped_input)**2, dim=(0,2,3)) / torch.sum(torch.abs(clamped_input), dim=(0,2,3))
#             # print('running_hoyer_thr: {}'.format(self.running_hoyer_thr))
            
#         # 
#         input = Spike_func.apply(input, hoyer_thr, x_thr_scale, self.spike_type, layer_index>=start_spike_layer)
#         # input = Spike_func.apply(input, hoyer_thr, x_thr_scale, self.spike_type, (layer_index>=13 and layer_index<=39))
#         return input

#     def extra_repr(self):
#         return (
#             "{num_features}, eps={eps}, momentum={momentum}, spike_type={spike_type}, "
#             "track_running_stats={track_running_stats}".format(**self.__dict__)
#         )
#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if (version is None or version < 2) and self.track_running_stats:
#             # at version 2: added num_batches_tracked buffer
#             #               this should have a default value of 0
#             num_batches_tracked_key = prefix + "num_batches_tracked"
#             if num_batches_tracked_key not in state_dict:
#                 state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

#         super(HoyerBiAct, self)._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )

class Spike_func(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, hoyer_thr, x_thr_scale=1.0, spike_type='sum', if_spike=True):
        ctx.save_for_backward(input)
        out = torch.clamp(input, min=0.0, max=1.0)
        # out = input
        # if torch.sum(torch.abs(out)) > 0:
        #     hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
        # else:
        #     hoyer_thr = 1.0
        ctx.if_spike = if_spike
        # print('input shape: {}, hoyer thr shape: {}, x_thr_scale: {}'.format(input.shape, hoyer_thr, x_thr_scale))
        if spike_type != 'cw':
            if if_spike:
                out[out < x_thr_scale*hoyer_thr] = 0.0
            # print('out shape: {}, x scale: {}, hoyer_thr: {}'.format(out.shape, x_thr_scale, hoyer_thr))
            out[out >= x_thr_scale*hoyer_thr] = 1.0
        else:
            if if_spike:
                out[out<x_thr_scale*hoyer_thr[None, :, None, None]] = 0.0
            out[out>=x_thr_scale*hoyer_thr[None, :, None, None]] = 1.0 
            # out[out<0.1*x_thr_scale*hoyer_thr[None, :, None, None]] = 0.0
            # out[out>=0.9*x_thr_scale*hoyer_thr[None, :, None, None]] = 1.0 
                    
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > 0] = 1.0
        # only for
        grad_inp[input > 2.0] = 0.0

        # grad_scale = 0.5 if ctx.if_spike else 1.0
        grad_scale = 1.0
    

        return grad_scale*grad_inp*grad_input, None, None, None, None


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        # real_weights = self.weight.view(self.shape)
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # #print(scaling_factor, flush=True)
        # scaling_factor = scaling_factor.detach()
        # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        # binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # #print(binary_weights, flush=True)
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        k = 6
        real_weights = self.weight.view(self.shape)
        Max=torch.max(real_weights)
        Min=torch.min(real_weights)
        if Max<-Min:
            Max=-Min
        Digital=torch.round(((2**k)-1)*real_weights/Max)
        quan_weights=Max*Digital/((2**k)-1)
        self.weight = quan_weights
        y = F.conv2d(x, quan_weights, stride=1, padding=1)

        return y

def hardBinaryConvForward(x, conv):
    real_weights = conv.weight
    # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
    # scaling_factor = scaling_factor.detach()
    # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
    # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
    # binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
    #print(binary_weights, flush=True)
    k = 3
    Max=torch.max(real_weights)
    Min=torch.min(real_weights)
    if Max<-Min:
        Max=-Min
    Digital=torch.round(((2**k)-1)*real_weights/Max)
    quan_weights=Max*Digital/((2**k)-1)
    y = F.conv2d(x, quan_weights, stride=1, padding=1)
    return y

class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, weight_quantize):
        k = weight_quantize 
        Max=torch.max(x)
        Min=torch.min(x)
        if Max<-Min:
            Max=-Min
        Digital = torch.round(((2**k)-1)*x/Max)
        x = Max*Digital/((2**k)-1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

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


class Threshold_relu(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        relu = nn.ReLU()
        # out = torch.zeros_like(input).cuda()
        out = relu(input-min_thr_scale)
        # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
        out[input >= max_thr_scale] = 1.0
        ctx.x_scale = x_thr_scale
        if layer_index >= start_spike_layer:
            out[out < ctx.x_scale*max_thr_scale] = 0.0
        # ctx.x_scale = 1.0
        out[out >= ctx.x_scale*max_thr_scale] = 1.0
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

        return grad_inp*grad_input, None, None, None, None, None, None
        # return grad_inp*grad_input, grad_threshold*grad_inp*grad_thr, None, None, None, None
class Threshold_mean(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        out = relu(input-0.0)
        out[input >= 1.0] = 1.0
        # 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
        if layer_index >= 44:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
        else:
            hoyer_thr = torch.mean(torch.sum((out)**2, dim=(1,2,3)) / torch.sum(torch.abs(out), dim=(1,2,3)))
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale*hoyer_thr, max_thr_scale*hoyer_thr
        ctx.x_scale = x_thr_scale
        # ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        if layer_index >= start_spike_layer:
            out[out <= ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        # out[out >= max_thr_scale*hoyer_thr] = 1.0
        # out[out <= min_thr_scale*hoyer_thr] = 0.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        epoch = ctx.epoch
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        # grad_inp[input > 0] = 1.0
        # grad_inp[input > 2*ctx.hoyer_thr] = 0.0
        # grad_inp[input > ctx.min_thr_scale] = 1.0
        # grad_inp[input > (2.0*ctx.max_thr_scale-ctx.min_thr_scale)] = 0.0
        # grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0
        grad_inp[input > (-1.0*0.0)] = 1.0
        grad_inp[input > (2.0*1.0)] = 0.0

        return grad_inp*grad_input, None, None, None, None, None, None
class Threshold_sum(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        out = relu(input-min_thr_scale)
        out[input >= max_thr_scale] = 1.0

        hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
        
        ctx.x_scale = x_thr_scale
        ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        if layer_index >= start_spike_layer:
            out[out < ctx.x_scale*hoyer_thr] = 0.0
        out[out >= ctx.x_scale*hoyer_thr] = 1.0
        
        # if layer_index >= start_spike_layer:
        #     out[out >= ctx.x_scale*hoyer_thr] = 1.0
        # out[out < ctx.x_scale*hoyer_thr] = 0.0
        
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
        # grad_inp[input > 2.0*ctx.hoyer_thr-1.0]=1.0
        # grad_inp[input > (1.0*ctx.max_thr_scale)] = 0.0

        return grad_inp*grad_input, None, None, None, None, None, None

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

# class Threshold_cw(torch.autograd.Function):
#     """
#     Here we use the piecewise-linear surrogate gradient as was done
#     in Bellec et al. (2018).
#     """
#     @staticmethod
#     def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
#         ctx.save_for_backward(input)
#         ctx.epoch = epoch
#         ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
#         ctx.layer_index = layer_index
#         relu = nn.ReLU()
#         # out = torch.zeros_like(input).cuda()
#         out = relu(input-min_thr_scale)
#         # 0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18
#         out[input >= max_thr_scale] = 1.0
#         if layer_index >= 44:
#             hoyer_thr = torch.mean(torch.sum((out)**2, dim=1) / torch.sum(torch.abs(out), dim=1))
#         else:
#             hoyer_cw = torch.sum((out)**2, dim=(2,3)) / torch.sum(torch.abs(out), dim=(2,3))
#             hoyer_cw = torch.nan_to_num(hoyer_cw, nan=0.0)
#             hoyer_cw = torch.mean(hoyer_cw, dim=0)
#             N,C,W,H = input.shape
#             hoyer_thr = torch.permute(hoyer_cw*(torch.ones(N,W,H,C)).cuda(), (0,3,1,2))
#             # print('mean: {}, sum: {}, cw: {}'.format(hoyer_mean, torch.mean(hoyer_sum), torch.mean(hoyer_thr)))
#             # print('layer: {}, hoyer_thr: {}'.format(layer_index, torch.mean(hoyer_thr)))
#         ctx.x_scale = x_thr_scale
#         # ctx.hoyer_thr = hoyer_thr
#         # hoyer_thr = 1.0
#         if layer_index >= start_spike_layer:
#             # print('1. layer: {}, hoyer_thr: {}'.format(layer_index, hoyer_thr))
#             out[out <= ctx.x_scale*hoyer_thr] = 0.0
#         out[out >= ctx.x_scale*hoyer_thr] = 1.0
        
#         # print('2. layer: {}, hoyer_cw: {}'.format(layer_index, hoyer_cw))
#         # exit()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
        
#         input,  = ctx.saved_tensors
#         epoch = ctx.epoch
#         grad_input = grad_output.clone()
#         grad_inp = torch.zeros_like(input).cuda()

#         grad_inp[input > (-1.0*ctx.min_thr_scale)] = 1.0
#         # grad_inp[input > (2.0*ctx.x_scale*ctx.hoyer_thr)] = 0.0
#         grad_inp[input > (2.0*ctx.max_thr_scale)] = 0.0

#         # print('layer: {}, grad_output_norm: {:.2f}, grad_inp_norm: {:.2f}, grad_out_norm: {:.2f}'.format(
#         #     ctx.layer_index, torch.norm(grad_output, p=2), torch.norm(grad_inp, p=2), torch.norm(grad_inp*grad_input, p=2)))

#         return grad_inp*grad_input, None, None, None, None, None, None

class ThrBiAct(nn.Module):
    def __init__(self, act_mode='sum') -> None:
        super(ThrBiAct, self).__init__()
        if act_mode == 'sum':
            self.F_ThrBiAct = Threshold_sum
        elif act_mode == 'mean':
            self.F_ThrBiAct = Threshold_mean
        elif act_mode == 'fixed':
            self.F_ThrBiAct = Threshold_relu
        elif act_mode == 'cw':
            self.F_ThrBiAct = Threshold_cw
        else:
            raise RuntimeError('invalid act_mode')
    def forward(self, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        return self.F_ThrBiAct.apply(input, epoch, min_thr_scale, max_thr_scale, x_thr_scale, layer_index, start_spike_layer)

class SubThr_sum(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    @staticmethod
    def forward(ctx, input, epoch, min_thr_scale=0.0, max_thr_scale=1.0, layer_index=0, x_thr_scale=1.0, start_spike_layer=0, scores=0.6):
        ctx.save_for_backward(input)
        ctx.epoch = epoch
        ctx.min_thr_scale, ctx.max_thr_scale = min_thr_scale, max_thr_scale
        ctx.layer_index = layer_index
        relu = nn.ReLU()
        out = relu(input-min_thr_scale)
        out[input >= max_thr_scale] = 1.0

        hoyer_thr = torch.sum((out)**2) / torch.sum(torch.abs(out))
        
        ctx.hoyer_thr = hoyer_thr
        # hoyer_thr = 1.0
        # for i in range(len(scores)):
        #     out_channel = out[:,i,:,:]
        #     out_channel[out_channel<scores[i]*hoyer_thr] = 0.0
        #     out_channel[out_channel>=scores[i]*hoyer_thr] = 1.0
        N,C,W,H = input.shape
        hoyer_cw = torch.permute((torch.ones(N,W,H,C).cuda())*scores*hoyer_thr, (0,3,1,2))
        out[out<hoyer_cw] = 0.0
        out[out>=hoyer_cw] = 1.0
        # if layer_index >= start_spike_layer:
        #     out[out < scores*hoyer_thr] = 0.0
        # out[out >= scores*hoyer_thr] = 1.0
        
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
        grad_score =  -1e3*torch.mean(grad_inp*grad_input, dim=(0,2,3)) / ctx.hoyer_thr
        # print('shape: {}, grad: {}, max: {}'.format(grad_output.shape, grad_score, torch.max(grad_score)))
        return grad_inp*grad_input, None, None, None, None, None, None, grad_score

# class GetActMask(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, scores, act_out):
#         mask = torch.mean(act_out.clone(), dim=0)
#         print('act_out shape: {}, ==1.0: {}, > 0.6: {}, > 0.4: {}, > 0.2: {}, >0.0: {}'.format(
#             mask.shape, torch.sum(mask==1.0), torch.sum(mask>0.6), torch.sum(mask>0.4), torch.sum(mask>0.2),torch.sum(mask>0.0)
#         ))
#         ctx.save_for_backward(act_out)
        
#         return act_out*scores
    
#     def backward(ctx, g):
#         mask, = ctx.saved_tensors
#         return g*mask, None

# class GetActMask(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, scores, act_out):
#         k=0.6
#         i=0.8
#         N  = act_out.shape[0]
#         mask_one = torch.ones_like(act_out).cuda()
#         mask_one[act_out<1.0] = 0.0
#         mask = scores.clone()
#         _, idx = scores.flatten().sort()
#         j = int((1 - k) * scores.numel())
#         ji = int((1 - i) * scores.numel())
#         # flat_out and out access the same memory. switched 0 and 1
#         flat_mask = mask.flatten()
#         flat_mask[idx[:j]] = 0
#         # flat_mask[idx[ji:j]] = 0.5
#         flat_mask[idx[j:]] = 1.0
#         out = torch.sign(act_out) * mask 
#         # mask[torch.mean(act_out, dim=0) > 1.0/N] = 1.0
#         ctx.save_for_backward(mask)
#         return out
    
#     def backward(ctx, g):
#         mask, = ctx.saved_tensors
#         # print('mask grad: {}, out grad: {}, total grad: {}'.format(torch.sum(mask), torch.sum(g), torch.sum(g*mask)))
#         return g*mask, None

class SubBiAct(nn.Module):
    def __init__(self, act_size, bit=1, act_mode='sum') -> None:
        super(SubBiAct, self).__init__()
        self.scores = nn.Parameter(torch.ones(act_size[0]))
        nn.init.constant_(self.scores, 0.5)
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.bit = bit
    @property
    def clamped_scores(self):
        # For unquantized activations
        return self.scores.abs()

    def forward(self, input, epoch, min_thr_scale=0, max_thr_scale=1, x_thr_scale=1.0, layer_index=0, start_spike_layer=0):
        # out = super().forward(out, epoch, min_thr_scale, max_thr_scale, layer_index, start_spike_layer)
        out = SubThr_sum.apply(input, epoch, min_thr_scale, max_thr_scale, x_thr_scale, layer_index, start_spike_layer, self.clamped_scores)
        # mask = GetActMask.apply(self.clamped_scores, input)
        # out = Threshold_sum.apply(input, epoch, min_thr_scale, max_thr_scale, layer_index, start_spike_layer)
        # out = GetActMask.apply(self.clamped_scores, out)
        return out

class BiConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):

        w = torch.sign(self.weight)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x