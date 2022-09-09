# import torch, detectron2
# # import mmdet
# # import mmcv
# TORCH_VERSION = torch.__version__
# CUDA_VERSION = torch.version.cuda
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

# print(mmcv.__version__)
# print(mmdet.__version__)

# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

'''
fuser -v /dev/nvidia*|awk -F " " '{print $0}' >/tmp/pid.file
while read pid ; do kill -9 $pid; done </tmp/pid.file
'''

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

