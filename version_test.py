import torch, detectron2
# import mmdet
# import mmcv
TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# print(mmcv.__version__)
# print(mmdet.__version__)

# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html