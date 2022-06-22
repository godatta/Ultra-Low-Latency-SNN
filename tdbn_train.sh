python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
--learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 0.0005 --momentum 0.9 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
--hoyer_decay 1e-8 --net_mode 'cut_2' --log --use_wandb \
--act_mode 'mean' --bn_type 'fake' --hoyer_type 'sum' --start_spike_layer 50 \
--description  'train tdbn with fake tdbn, with act loss, spike when < scaled thr, cut 0.2, 0.8, original (0~2) bp' \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206131013.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206122147.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206091914.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206111023.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206032334.pth'
# 'trained_models_ann/ann_vgg16_cifar10_202206050043.pth'

# --description 'train tdbn with true tdbn, with act loss, v1 bp' 

# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 4e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0 --momentum 0.95 --amsgrad True --devices 0 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 0 --net_mode 'ori' --log \
# --act_mode 'spike' --bn_type 'tdbn' --hoyer_type 'sum' --start_spike_layer 0 \
# --description 'train tdbn with fake tdbn, with act loss, force spike for > hoyer_thr, mean hoyer thr, original bp' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206210004.pth'


# 'train tdbn with fake tdbn, without act loss, force spike when > sum hoyer thr, scale=0.8, new version channelwise scale factor'
# 2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42, 44, 46
