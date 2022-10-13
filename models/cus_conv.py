import torch
import torch.nn as nn

class customConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('identity_kernel', torch.ones(out_channels, in_channels, *kernel_size))
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=True)
        with torch.no_grad():
            self.weights.data.normal_(0.0, 0.8)

    def forward(self, img):
        
        b, c, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        p00 = 0.0
        p01 = -0.000287
        p10 = 0.0
        p11 = 0.266
        p20 = 0.0
        p21 = -0.1097
        p30 = 0.0

        img_unf = nn.functional.unfold(img, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding).transpose(1, 2).contiguous()
        self.identity_kernel = self.identity_kernel.contiguous()
        identity_weights = self.identity_kernel.view(self.identity_kernel.size(0), -1).contiguous()
        self.weights = self.weights.contiguous()
        weights = self.weights.view(self.weights.size(0), -1).contiguous()

        # f0 = (p00 + torch.zeros_like(img_unf)).matmul(identity_weights.t())
        # f1 = (p10 * (img_unf - 0.5)).matmul(identity_weights.t())
        # f2 = (p01 * torch.ones_like(img_unf)).matmul(weights.t())
        # f3 = (p20 * torch.pow(img_unf - 0.5, 2)).matmul(identity_weights.t())
        # f4 = (p11 * (img_unf - 0.5)).matmul(weights.t())
        # f5 = (p30 * torch.pow(img_unf - 0.5, 3)).matmul(identity_weights.t())
        # f6 = (p21 * torch.pow(img_unf - 0.5, 2)).matmul(weights.t())
        # f = (f0 + f1 + f2 + f3 + f4 + f5 + f6).transpose(1, 2)

        f = ((p00 + torch.zeros_like(img_unf) +
             p10 * (img_unf) +
             p20 * torch.pow(img_unf, 2) +
             p30 * torch.pow(img_unf, 3)).matmul(identity_weights.t()) + \
            (p01 * torch.ones_like(img_unf) +
             p11 * (img_unf) +
             p21 * torch.pow(img_unf, 2)
             ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        
        out_xshape = int((h-self.kernel_size[0]+2*self.padding)/self.stride) + 1
        out_yshape = int((w-self.kernel_size[1]+2*self.padding)/self.stride) + 1
        #out = f.contiguous()
        out = f.view(b, self.out_channels, out_xshape, out_yshape)#.contiguous()
        out = out/(3*self.kernel_size[0]*self.kernel_size[1])
        return out
    def extra_repr(self):
        return (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}".format(**self.__dict__)
        )