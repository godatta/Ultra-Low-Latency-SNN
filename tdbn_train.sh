# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --use_wandb --pool_pos 'before_relu' \
# --act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 \
# --description  'train tdbn with bn, with act loss, spike when > sum_hoyer replace thr, pool directly after conv' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131013.pth'
# 93.25
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth' 
# 93.95
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206111023.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206050043.pth'

# --description 'train tdbn with true tdbn, with act loss, v1 bp' 

python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
--learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 0 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
--hoyer_decay 1e-8 --net_mode 'ori' --log --use_wandb \
--act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 \
--description  'train tdbn with bn, with act loss, spike when > sum_hoyer replace thr, pool directly after conv, scale=0.8' \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'


# 'train tdbn with fake tdbn, without act loss, force spike when > sum hoyer thr, scale=0.8, new version channelwise scale factor'
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
