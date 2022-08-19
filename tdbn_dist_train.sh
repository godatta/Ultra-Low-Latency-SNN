# torchrun --nproc_per_node=4 ann.py --dataset IMAGENET --batch_size 16 --architecture VGG16 \
# --learning_rate 1e-02 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0,1,2,3 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --thr_decay 0 --reg_type 1 --hoyer_decay 1e-8 --net_mode 'ori' --act_type 'tdbn' --log --use_wandb \
# --act_mode 'fixed' --bn_type 'bn' --hoyer_type 'sum' --start_spike_layer 50 \
# --description 'test resnet' 
# torchrun --nproc_per_node=2 ann_imagenet.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 \
# --learning_rate 2e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0001 --momentum 0.95 --amsgrad True --devices 0,1 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --net_mode 'ori' --log --pool_pos 'before_relu' --bn_type 'bn' \
# --act_mode 'cw' --hoyer_type 'sum' --hoyer_decay 1e-8 --start_spike_layer 0 --x_thr_scale 0.618 --weight_quantize 0 \
# --description 'loss for output, cos lr, grad=0.5' --use_wandb --warmup --lr_decay 'cos'
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206181137.pth'

# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth' --use_wandb

torchrun --nproc_per_node=8 ann_imagenet.py --dataset IMAGENET --batch_size 64 --im_size 224 --architecture VGG16 \
--learning_rate 1e-3 --epochs 150 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer Adam --weight_decay 5e-6 --momentum 0.9 --amsgrad True --devices 0,1,2,3,4,5,6,7 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
--net_mode 'ori' --log --pool_pos 'before_relu' --bn_type 'bn' \
--act_mode 'cw' --hoyer_type 'sum' --hoyer_decay 1e-8 --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'imagenet test with wandb' --use_wandb --warmup