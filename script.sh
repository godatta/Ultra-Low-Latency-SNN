# Train an ANN for <architecture> on <dataset>
#python ann.py --dataset IMAGENET --batch_size 32 --architecture VGG16 --learning_rate 1e-2 --epochs 100 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 0,1,2 --seed 0 --kernel_size 3 --momentum 0.9 --relu_threshold 4.0 --weight_decay 1e-4 --log
python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-2 --epochs 2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.3 --devices 3 --relu_threshold 4.0 --momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --pretrained_ann 'ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture RESNET20 --learning_rate 1e-2 --epochs 2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.3 --devices 0 --relu_threshold 4.0 --momentum 0.95 --weight_decay 0.0005 --seed 0 --pretrained_ann 'trained_models_ann/ann_resnet20_cifar10_4.0.pth'
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-2 --epochs 2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.3 --devices 0 --relu_threshold 4.0 --momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --#pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
# Convert ANN to SNN and perform spike-based backpropagation
#python snn.py --dataset CIFAR100  --batch_size 256 --architecture RESNET20 --learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 2 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 1500 --devices 2 --default_threshold 1.0 --pretrained_ann '/home/gdatta/dietsnn_2021/code/trained_models_ann/ann_resnet20_cifar100_4.0_0.3resnet_trying_again_act_compute.pth' --log
#python snn.py --dataset CIFAR100 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 2 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 12 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 1500 --devices 2 --default_threshold 1.0 --pretrained_ann 'ann_vgg16_cifar100_4.0normal.pth' --test_only
#python ann.py --dataset IMAGENET --batch_size 128 --architecture VGG16 --learning_rate 1e-2 --epochs 100 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.2 --devices 2 --seed 0 --kernel_size 3 --momentum 0.9 --relu_threshold 4.0 --log
#python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 300 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 1 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 1500 --devices 0 --default_threshold 1.0 --pretrained_ann '/home/gdatta/dietsnn_2021/code/trained_models_ann/ann_vgg16_cifar10_4.0.pth' --log 

# orinal
# python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 200 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 1 --leak 1.0 --scaling_factor 0.5 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 100 --devices 0 --default_threshold 1.0 --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth' --log

# train snn.py
# python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 10 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 5 --leak 1.0 --scaling_factor 0.5 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 100 --devices 0 --default_threshold 1.0 --log --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'

#python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --log --epochs 200 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 1 --leak 1.0 --scaling_factor 0.5 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 1500 --devices 3 --default_threshold 1.0 --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'
# python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 200 --lr_interval '0.60 0.80 0.90' --lr_reduce 5 --timesteps 1 --leak 1.0 --scaling_factor 0.5 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 100 --devices 1 --default_threshold 1.0 --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'

# train ann
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-2 --epochs 200 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.3 --devices 0 --relu_threshold 4.0 --momentum 0.95 --weight_decay 0.0005 --seed 0

# test ann
# python ann.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-2 --epochs 2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD --dropout 0.3 --devices 0 --relu_threshold 4.0 --momentum 0.95 --weight_decay 0.0005 --seed 0 --test_only --pretrained_ann 'trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth'

# prune snn
# python snn.py --dataset CIFAR10 --batch_size 128 --architecture VGG16 --learning_rate 1e-4 --epochs 45 --lr_interval '0.60 0.80 0.90' --lr_reduce 1 --timesteps 5 --leak 1.0 --scaling_factor 1.0 --optimizer Adam --weight_decay 0.000 --momentum 0.9 --amsgrad True --dropout 0.1 --train_acc_batches 100 --devices 0 --default_threshold 1.0 --pretrained_snn 'trained_snn_models/snn_vgg16_cifar10_5_2022032511.pth' --log

