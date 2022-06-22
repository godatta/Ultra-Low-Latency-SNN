# torchrun --nproc_per_node=4 ann_dist.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 \
# --learning_rate 8e-04 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0,1,2,3 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --thr_decay 0 --reg_type 1 --hoyer_decay 1e-8 --net_mode 'ori' --act_type 'tdbn' --log \
# --act_mode 'spike' --bn_type 'fake' --hoyer_type 'sum' --start_spike_layer 0 --conv_type 'dy' \
# --description 'train tdbn with fake tdbn, spike hoyer_thr, mean' \
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth'

torchrun --nproc_per_node=4 ann_dist.py --dataset CIFAR10 --batch_size 64 --architecture VGG16 \
--learning_rate 2e-04 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 0.0 --momentum 0.9 --amsgrad True --devices 0,1,2,3 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
--thr_decay 0 --reg_type 1 --hoyer_decay 0 --net_mode 'ori' --act_type 'tdbn' --log \
--act_mode 'mean' --bn_type 'fake' --hoyer_type 'sum' --start_spike_layer 0 --conv_type 'dy' \
--description 'train tdbn with fake tdbn, spike hoyer_thr, mean' \
--pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth'