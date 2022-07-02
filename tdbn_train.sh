# 1.
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-2 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --use_wandb \
# --act_mode 'cw' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 --pooling_type 'max' \
# --description 'train tdbn with bn, with act loss, force 1 when > fixed_hoyer, pool before relu, maxpool' 
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth'

# 93.92 --use_wandb
# 'trained_models_ann/ann_vgg16_cifar10_202206241620.pth'
# 93.82
# 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 93.25
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth' 
# 93.95
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206111023.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206050043.pth'

# --description 'train tdbn with true tdbn, with act loss, v1 bp' 
# 2.
python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
--learning_rate 4e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer SGD --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
--hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --use_wandb \
--act_mode 'cw' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 --pooling_type 'avg' --x_thr_scale 1.0 \
--description 'train tdbn with bn, with act loss, spike when > cw_hoyer, pool directly after conv, set hoyer_thr like bn, avgpool' \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202207011610.pth'
# 

# 3. 
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0000 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --use_wandb \
# --act_mode 'cw' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 --x_thr_scale 0.618 \
# --description 'train tdbn with bn, with act loss, spike when > sum_hoyer, pool directly after conv, set hoyer_thr like bn' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'
# 'train tdbn with fake tdbn, without act loss, force spike when > sum hoyer thr, scale=0.8, new version channelwise scale factor'
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46

# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 1e-1 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --log --pool_pos 'before_relu' --sub_act_mask \
# --act_mode 'sum' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 0 \
# --description 'train tdbn with bn, with act loss, force 1 when > sum_hoyer, pool directly after conv, test mask, scale' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206250007.pth'