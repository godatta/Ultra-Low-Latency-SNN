# 1.
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0001 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-9 --net_mode 'ori' --pool_pos 'before_relu' --log --use_wandb --use_hook \
# --act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 --x_thr_scale 0.618 --weight_quantize 0 \
# --description 'test if calculate hoyer loss before spike, small hoyer loss' 

# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth' --use_wandb --log

# 1. RESNET20 + CIFAR10
# python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture mobilenetv3_small \
# --learning_rate 1e-2 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0 --conv_dropout 0 \
# --hoyer_decay 0 --net_mode 'ori' --pool_pos 'before_relu'  --use_hook  --reg_thr --log \
# --spike_type 'sum' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'mobilenetv3 test' --warmup 1000 --lr_decay 'step' --use_wandb

# put bn after add res, --log --use_wandb 
torchrun --nproc_per_node=7 ann.py --dataset IMAGENET --batch_size 64 --im_size 224 --architecture RESNET18 \
--learning_rate 5e-4 --epochs 150 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 0.0005 --momentum 0.9 --amsgrad True --devices 0,1,2,3,4,5,6 --seed 0 --linear_dropout 0 --conv_dropout 0 \
--net_mode 'ori' --log --pool_pos 'before_relu' --bn_type 'bn' \
--spike_type 'sum' --loss_type 'sum' --hoyer_decay 2e-9 --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'test resnet50 sum, forward 2.0, , downsaple 2.0 with bn for bottleneck' --use_hook --reg_thr --warmup 1000 --lr_decay 'step'

# --description 'resnet20 test spike->conv->bn without dropout, layer wise hoyer_reg '  --use_wandb

# 93.92 --use_wandb --use_reg --use_hook
# 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth'
# 93.82
# 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 93.25
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth' 
# 93.95
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 93.93
# 'trained_models_ann/ann_vgg16_cifar10_202207021255.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206111023.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206050043.pth'

# --description 'train tdbn with true tdbn, with act loss, v1 bp' 
# 2.
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 4e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --use_wandb \
# --act_mode 'cw' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 --pooling_type 'max' --x_thr_scale 1.0 \
# --description 'train tdbn with bn, with act loss, spike when > cw_hoyer, pool directly after conv, set hoyer_thr like bn, maxpool, <13,>39,0.5*[0,2]bp' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202207021255.pth'
# 

# 3. IMAGENET
# python ann.py --dataset IMAGENET --batch_size 4 --im_size 224 --architecture VGG16 \
# --learning_rate 1e-2 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --warmup \
# --act_mode 'fixed' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'imagenet test without wandb'
# --description 'train tdbn with bn, with act loss, spike when > sum_hoyer, pool directly after conv, set hoyer_thr like bn, maxpool, grad scale=0.5, no bias, only weight decay for weights' 
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202207060058.pth'
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46 --use_wandb

# train quantize
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 600 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0001 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --use_wandb \
# --act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 --x_thr_scale 0.618 --weight_quantize 2 \
# --description 'quan 2 bit, test new quantize' 
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202207062203.pth'

# 93.15
# 'trained_models_ann/ann_vgg16_cifar10_202207062203.pth'

# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-1 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --sub_act_mask \
# --act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 \
# --description 'train tdbn with bn, with act loss, force 1 when > sum_hoyer, pool directly after conv, test mask, scale' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'