
# python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture vgg16_light \
# --learning_rate 1e-4 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 1 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu'  --if_set_0 \
# --spike_type 'cw' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'both clip' --warmup 0 --lr_decay 'step' --use_hook --log 
# --pretrained_ann 'trained_models_ann/ann_vgg16_light_cifar10_202209141451.pth'

# train VGG16 0,2
# python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture vgg16_light_multi_steps \
# --learning_rate 1e-4 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 1 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --hoyer_decay 0 --net_mode 'ori' --pool_pos 'before_relu' --if_set_0 \
# --spike_type 'fixed' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 0.5 --weight_quantize 0 \
# --description 'EE641, T=5, thr scale: 0.5, grad scale: 0.5' --warmup 0 --lr_decay 'step' --use_hook --log --time_step 5 --reg_thr  --use_wandb 
# --pretrained_ann 'saved_models/ann_vgg16_light_multi_steps_cifar10_202211161716.pth'


# train ResNet18 --use_wandb --use_hook echo $CUDA_VISIBLE_DEVICES
# python ann.py --dataset CIFAR10 --batch_size 128 --im_size 32 --architecture resnet18 \
# --learning_rate 1e-1 --epochs 600 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 1 --seed 0 --linear_dropout 0 --conv_dropout 0 \
# --hoyer_decay 0 --net_mode 'ori' --pool_pos 'before_relu' \
# --spike_type 'fixed' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'both with clip' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook --use_wandb 
# --description 'fixed hoyer threshold=0.5' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook
# --description 'fixed threshold=1.0, without clamp 1.0, grad>2*thr=0.0, grad scale=1.0' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook

# # multistep resnet18
python ann.py --dataset CIFAR10 --batch_size 256 --im_size 32 --architecture resnet18_multi_steps \
--learning_rate 2e-1 --epochs 600 --lr_interval '0.50 0.70 0.85' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 6,7 --seed 0 --linear_dropout 0 --conv_dropout 0 \
--hoyer_decay 0 --net_mode 'ori' --pool_pos 'before_relu' \
--spike_type 'fixed' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 0.5 --weight_quantize 0 \
--description 'EE641, T=1, thr scale: 0.5, grad scale: LIF' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook --time_step 1 --use_wandb \
# --pretrained_ann 'saved_models/ann_resnet18_multi_steps_cifar10_202212041432.pth'
# saved_models/ann_resnet18_multi_steps_cifar10_202211171750.pth
# saved_models/ann_resnet18_multi_steps_cifar10_202211191119.pth