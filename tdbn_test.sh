python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
--learning_rate 1e-4 --epochs 1 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 \
--optimizer Adam --devices 0 --relu_threshold 1.0 \
--momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --linear_dropout 0.1 --conv_dropout 0.1 \
--hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu' --test_type 'v1' \
--act_mode 'sum' --bn_type 'bn' --start_spike_layer 0 --hoyer_type 'sum' --x_thr_scale 0.618 --weight_quantize 6 \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202207062203.pth'
# 93.92 
# 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth'
# 93.82
# 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 93.25
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth' 
# 93.15
# 'trained_models_ann/ann_vgg16_cifar10_202207062203.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206151759.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131013.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131841.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206072304.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'

# 'trained_models_ann/ann_vgg16_cifar10_202206151759.pth'
# 2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42, 44, 46 
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
# --conv_type 'dy'  --get_layer_output --get_scale 
# 202206131035 :
# 92.62 -> 92.63 -> 92.61 -> 92.65 -> 92.47 -> 92.32 -> 92.35 -> 91.73 --all--> 82.54


# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 1 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 \
# --optimizer Adam --devices 0 --relu_threshold 4.0 \
# --momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --linear_dropout 0.3 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' \
# --act_mode 'mean' --bn_type 'fake' --start_spike_layer 0 --hoyer_type 'sum' --conv_type 'dy' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth'