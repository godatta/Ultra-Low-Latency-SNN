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
        min_thr_scale=0.0, max_thr_scale=1.0, x_thr_scale=1.0, if_spike=True, if_set_0=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HoyerBiAct, self).__init__()
        self.num_features   = num_features if spike_type == 'cw' else 1
        self.eps            = eps
        self.momentum       = momentum
        self.spike_type     = spike_type
        self.track_running_stats = track_running_stats
        self.threshold      = nn.Parameter(torch.tensor(1.0))
        # self.threshold    = 1.0
        self.min_thr_scale  = min_thr_scale
        self.max_thr_scale  = max_thr_scale
        self.x_thr_scale    = x_thr_scale
        self.if_spike       = if_spike  
        self.if_set_0       = if_set_0
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
            # clamped_input = torch.clamp((input).clone().detach(), min=0.0)
            clamped_input = torch.clamp((input).clone().detach(), min=0.0, max=1.0)
            # if self.if_set_0:
            #     clamped_input[clamped_input >= 1.0] = 0.0

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
            
        out = Spike_func.apply(input, hoyer_thr, self.x_thr_scale, self.spike_type, self.if_spike)

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

class Spike_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, hoyer_thr, x_thr_scale=1.0, spike_type='sum', if_spike=True):
        ctx.save_for_backward(input)
        out = input.clone()
        ctx.hoyer_thr = hoyer_thr

        if spike_type != 'cw':
            out[out < x_thr_scale*hoyer_thr] = 0.0
            out[out >= x_thr_scale*hoyer_thr] = 1.0
        else:
            out[out<x_thr_scale*hoyer_thr[None, :, None, None]] = 0.0
            out[out>=x_thr_scale*hoyer_thr[None, :, None, None]] = 1.0 
             
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,  = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_inp = torch.zeros_like(input).cuda()

        grad_inp[input > 0] = 1.0
        grad_inp[input > 2.0] = 0.0
        # grad_inp= torch.abs(1.0-input.clone()) + 1.0
        # grad_inp[input < 0] = 0.0
        # grad_inp[input > 2.0] = 0.0
        # grad_inp[input > 2.0*ctx.hoyer_thr] = 0.0

        # grad_scale = 0.5 if ctx.if_spike else 1.0
        grad_scale = 0.5
    

        return grad_scale*grad_inp*grad_input, None, None, None, None


class HoyerBiAct_multi_step(HoyerBiAct):
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
    def __init__(self, **kwargs):
        super(HoyerBiAct_multi_step, self).__init__(**kwargs)
        self.mem = 0.0
        self.leak = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, input, T=1):
        self.mem = 0.0 if T == 1 else self.mem
        # calculate running estimates
        input = input / torch.abs(self.threshold)
        # input = torch.clamp(input, min=0.0, max=1.0)
        if self.training:
            clamped_input = torch.clamp((input).clone().detach(), min=0.0, max=1.0)
            if self.if_set_0:
                clamped_input[clamped_input >= 1.0] = 0.0

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
        self.mem = self.leak*self.mem + input 
        out = Spike_func.apply(self.mem, hoyer_thr, self.x_thr_scale, self.spike_type, self.if_spike)
        return out