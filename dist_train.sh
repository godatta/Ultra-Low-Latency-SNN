# torchrun --nproc_per_node=4 ann.py --dataset IMAGENET --batch_size 16 --architecture VGG16 \
# --learning_rate 1e-02 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0005 --momentum 0.95 --amsgrad True --devices 0,1,2,3 --seed 0 --linear_dropout 0.3 --conv_dropout 0.1 \
# --thr_decay 0 --reg_type 1 --hoyer_decay 1e-8 --net_mode 'ori' --act_type 'tdbn' --log --use_wandb \
# --spike_type 'fixed' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 50 \
# --description 'test resnet' 
# torchrun --nproc_per_node=2 ann.py --dataset CIFAR10 --im_size 32 --batch_size 128 --architecture RESNET34 \
# --learning_rate 2e-1 --epochs 400 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 0,1 --seed 0 --linear_dropout 0 --conv_dropout 0 \
# --net_mode 'ori' --log --pool_pos 'before_relu' --bn_type 'bn' \
# --spike_type 'cw' --loss_type 'sum' --hoyer_decay 1e-8 --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'resnet 34 cw' 
# --use_wandb --warmup 
# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206181137.pth' --lr_decay 'cos'

# --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_202206211113.pth' --use_wandb

# torchrun --nproc_per_node=8 ann.py --dataset IMAGENET --batch_size 64 --im_size 224 --architecture vgg16_light \
# --learning_rate 1e-4 --epochs 150 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer Adam --weight_decay 0.0005 --momentum 0.9 --amsgrad True --devices 0,1,2,3,4,5,6,7 --seed 0 --linear_dropout 0.1 --conv_dropout 0.1 \
# --net_mode 'ori' --log --pool_pos 'before_relu' --bn_type 'bn' --if_set_0 \
# --spike_type 'cw' --loss_type 'sum' --hoyer_decay 1e-8 --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'train a spike vgg16' --use_wandb --warmup 1000 --lr_decay 'step'

# torchrun --nproc_per_node=2 ann.py --dataset CIFAR10 --batch_size 256 --im_size 32 --architecture resnet18_multi_steps \
# --learning_rate 2e-1 --epochs 600 --lr_interval '0.50 0.70 0.85' --lr_reduce 5 --relu_threshold 1.0 \
# --optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 6,7 --seed 0 --linear_dropout 0 --conv_dropout 0 \
# --hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu' \
# --spike_type 'cw' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
# --description 'dist resnet20 t=2' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook --time_step 2

# echo $CUDA_VISIBLE_DEVICES 2,5,  6,7
torchrun --nproc_per_node=2 --master_port 29500 ann.py --dataset CIFAR10 --batch_size 64 --im_size 32 --architecture mobilenet \
--learning_rate 1e-1 --epochs 400 --lr_interval '0.50 0.70 0.85' --lr_reduce 5 --relu_threshold 1.0 \
--optimizer SGD --weight_decay 0.0001 --momentum 0.9 --amsgrad True --devices 6,7 --seed 0 --linear_dropout 0 --conv_dropout 0 \
--hoyer_decay 1e-8 --net_mode 'ori' --pool_pos 'before_relu' \
--spike_type 'cw' --bn_type 'bn' --loss_type 'sum' --start_spike_layer 0 --x_thr_scale 1.0 --weight_quantize 0 \
--description 'dist mobilenet t=2' --warmup 0 --lr_decay 'step' --reg_thr --log --use_hook --time_step 1