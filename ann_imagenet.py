import argparse
from models.vgg_tunable_threshold_tdbn_imagenet import VGG_TUNABLE_THRESHOLD_tdbn_imagenet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb
import sys
import datetime
import os
import numpy as np
import cv2
from tqdm import tqdm
from math import cos, pi
from utils.net_utils import *
# from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.utils.data.distributed import DistributedSampler
#from data_prep import *
#from self_models import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        # print('output: {}, target: {}, pred: {}, correct: {}, res: {}'.format(output.shape, target.shape, pred.shape, correct.shape, res))
        # exit()
        return res

def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.learning_rate * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.learning_rate * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.learning_rate * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.learning_rate * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.learning_rate * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_mac(model, dataset):

    if dataset.lower().startswith('cifar'):
        h_in, w_in = 32, 32
    elif dataset.lower().startswith('image'):
        h_in, w_in = 224, 224
    elif dataset.lower().startswith('mnist'):
        h_in, w_in = 28, 28

    macs = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Conv2d):
            c_in    = l.in_channels
            k       = l.kernel_size[0]
            h_out   = int((h_in-k+2*l.padding[0])/(l.stride[0])) + 1
            w_out   = int((w_in-k+2*l.padding[0])/(l.stride[0])) + 1
            c_out   = l.out_channels
            mac     = k*k*c_in*h_out*w_out*c_out
            if mac == 0:
                pdb.set_trace()
            macs.append(mac)
            h_in    = h_out
            w_in    = w_out
            print('{}, Mac:{}'.format(name, mac))
        if isinstance(l, nn.Linear):
            mac     = l.in_features * l.out_features
            macs.append(mac)
            print('{}, Mac:{}'.format(name, mac))
        if isinstance(l, nn.AvgPool2d):
            h_in    = h_in//l.kernel_size
            w_in    = w_in//l.kernel_size
    print('{:e}'.format(sum(macs)))
    exit()

def train(epoch, loader):

    global learning_rate
    
    losses = AverageMeter('Loss')
    act_losses = AverageMeter('Loss')
    total_losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    top5   = AverageMeter('Acc@5')
    
    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
  
    model.train() # this is important, cannot remove
    
    with tqdm(loader, total=len(loader)) as t:
        for batch_idx, (data, target) in enumerate(t):
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
                    
            optimizer.zero_grad()

            output, _, _, act_out = model(data)
            loss = F.cross_entropy(output,target)

            data_size = data.size(0)
            act_loss = hoyer_decay*act_out
            total_loss = loss + act_loss
            total_loss.backward(inputs = list(model.parameters()))
            
            optimizer.step()       
            
            losses.update(loss.item(),data_size)
            act_losses.update(act_loss, data_size)
            total_losses.update(total_loss.item(), data_size)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1.update(prec1.item(), data_size)
            top5.update(prec5.item(), data_size)


            # torch.cuda.empty_cache()
            if local_rank==0 and ((epoch == 1 and batch_idx < 5) or (dataset == 'IMAGENET' and batch_idx%100==1)):
                f.write('\nbatch: {}, train_loss: {:.4f}, act_loss: {:.4f}, total_train_loss: {:.4f} '.format(
                batch_idx,
                losses.avg,
                act_losses.avg,
                total_losses.avg,
                ))
                f.write('top1_acc: {:.2f}, top5_acc: {:.2f}, time: {}'.format(
                top1.avg,
                top5.avg,
                datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
                ))
                if use_wandb:
                    wandb.log({
                        'loss': losses.avg,
                        'loss_act': act_losses.avg,
                        'total_loss': total_losses.avg
                    })
                    # for i in range(len(model.test_hoyer_thr)):
                    #     wandb.log({f'hoyer_thr_{i}': test_hoyer_thr[i]/batch_idx}, step=epoch)
                    wandb.log({'training_acc': top1.avg})
                    wandb.log({'top_5_acc': top5.avg})
             
        f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, act_loss: {:.4f}, total_train_loss: {:.4f} '.format(
                epoch,
                learning_rate,
                losses.avg,
                act_losses.avg,
                total_losses.avg,
                )
            )
        f.write('top1_acc: {:.4f}%, top5_acc: {:.4f}%, time: {}'.format(
                top1.avg,
                top5.avg,
                datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
                )
            )

def test(epoch, loader):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    top5   = AverageMeter('Acc@5')
    act_losses = AverageMeter('Loss') 
    total_losses = AverageMeter('Loss')
    hoyer_thr_per_batch = []

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        #dis = []
        total_output = {}
        plot_output = {}
        global max_accuracy, start_time

        for batch_idx, (data, target) in enumerate(tqdm(loader)):
        # for batch_idx, (data, target) in enumerate(loader):
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            output, _, _, act_out = model(data)

            loss = F.cross_entropy(output,target)
            data_size = data.size(0)

            act_loss = hoyer_decay*act_out
            total_loss = loss+act_loss

            act_losses.update(act_loss, data_size)
            losses.update(loss.item(), data_size)
            total_losses.update(total_loss.item(), data_size)

            prec1, prec5 = accuracy(output, target, topk=(1, 2 ))

            top1.update(prec1.item(), data_size)
            top5.update(prec5.item(), data_size)

        if not test_only and use_wandb:
            wandb.log({'test_acc': top1.avg}, step=epoch)
            wandb.log({'test_top5_acc': top5.avg}, step=epoch)
        # writer.add_scalar('Accuracy/test', top1.avg, epoch)
        # if (top1.avg>=max_accuracy) and top1.avg>0.88:
        if (top1.avg>=max_accuracy):
            max_accuracy = top1.avg
            # if not test_only:
            #     wandb.run.summary["best_accuracy"] = top1.avg
            state = {
                    'accuracy'      : max_accuracy,
                    'epoch'         : epoch,
                    'state_dict'    : model.state_dict() if gpu_nums==1 else model.module.state_dict(),
                    'optimizer'     : optimizer.state_dict()
            }
            try:
                os.mkdir('./trained_models_ann/')
            except OSError:
                pass
            
            # filename = './trained_models_ann/'+identifier+ '_epoch_' + str(epoch) + '_' + str(max_accuracy) + '.pth'
            filename = './trained_models_ann/'+identifier + '.pth'
            if not args.dont_save and not test_only:
                torch.save(state,filename)
        #dis = np.array(dis)
        f.write('\nEpoch: {}, best: {:.4f}, test_loss: {:.4f}, act_loss: {:.4f}, total_test_loss: {:.4f}, '.format(
            epoch,
            max_accuracy,
            losses.avg,
            act_losses.avg,
            total_losses.avg,
            )
        )
        f.write('top1_acc: {:.4f}, top5_acc: {:.4f}, time: {}\n'.format(
            top1.avg,
            top5.avg,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

def visualize(loader, to_path, visual_type=['directly', 'grad_cam'], num_imgs=100):

    model.eval()
    try:
        os.makedirs(to_path)
    except OSError:
        pass

    def minmax(x):
        return (x-np.min(x))/(1e-10+np.max(x)-np.min(x))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    name_list = []
    module_list = []
    for (name, module) in model.named_modules():
        if name.endswith('pool_weight'):
            module.register_forward_hook(get_activation(name))
            name_list.append(name)
            module_list.append(module)

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if batch_idx > num_imgs:
            break

        if torch.cuda.is_available() and args.gpu:
            data, target = data.to('cuda:2'), target.to('cuda:2')

        output = model(data)

        pred = output.max(1,keepdim=True)[1]
        (b, c, h, w) = data.shape
        img = data[0,:,:,:].detach().cpu().numpy().transpose([1,2,0])
        img = np.uint8(255 * minmax(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_name = f'img{batch_idx+1}'

        cv2.imwrite(os.path.join(f'/nas/home/fangc/Non-Local-Pooling/base/visualization/imgs/{img_name}.jpg'), img)


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):

        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32
        self.test_hoyer_thr = torch.tensor([0.0]*15)

    def forward(self, x, epoch=0):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x, self.threshold_out, self.relu_batch_num, act_out  = self.model_fp32(x, epoch)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x, self.threshold_out, self.relu_batch_num, act_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--description',            default='exp desc',         type=str,       help='description for the exp')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')

    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=0.000,              type=float,     help='weight_decay')
    parser.add_argument('--momentum',               default=0.9,                type=float,     help='mometum of optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad')
    parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
    parser.add_argument('--lr_interval',            default='0.45 0.70 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--lr_decay',               default='step',             type=str,       help='mode for learning rate decay')

    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100', 'IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('--im_size',                default=None,               type=int,       help='image size')


    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG4','VGG6','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-rthr','--relu_threshold', default=0.5,                type=float,     help='threshold value for the RELU activation')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--linear_dropout',         default=0.1,                type=float,     help='dropout percentage for linear layers')
    parser.add_argument('--conv_dropout',           default=0.1,                type=float,     help='dropout percentage for conv layers')

    parser.add_argument('--get_layer_output',       action='store_true',                        help='save the output of eavry layer')
    parser.add_argument('--use_init_thr',           action='store_true',                        help='use the inital threshold')
    parser.add_argument('--get_scale',              action='store_true',                        help='get the scale factors for every layer')
    parser.add_argument('--use_x_scale',            action='store_true',                        help='use the scale factors for every layer')
    # parser.add_argument('--reg_decay',              default=0.0001,             type=float,     help='weight decay for threshold loss (default: 0.001)')
    # parser.add_argument('--reg_type',               default=0,                  type=int,       help='regularization type: 0:None 1:L1 2:Hoyer 3:HS')
    parser.add_argument('--hoyer_decay',            default=0.0001,             type=float,     help='weight decay for regularizer (default: 0.001), original: act_decay')
    parser.add_argument('--net_mode',               default='ori',              type=str,       help='ori: original one, cut: cut the threshold')
    # parser.add_argument('--act_type',               default='spike',            type=str,       help='thr: tunable threshold, relu: relu with thr, spike: tunable spiking')
    # parser.add_argument('--thr_decay',              default=0.0001,             type=float,     help='weight decay for threshold loss (default: 0.001)')
    parser.add_argument('--hoyer_type',             default='mean',             type=str,       help='mean:, sum:, mask')
    parser.add_argument('--act_mode',               default='v1',               type=str,       help='fixed: threshold always is 1.0, sum: use sum hoyer as thr, channelwise(cw): use cw hoyer as thr ')
    parser.add_argument('--start_spike_layer',      default=50,                 type=int,       help='start_spike_layer')
    parser.add_argument('--bn_type',                default='bn',               type=str,       help='bn: , tdbn: , fake: the type of batch normalization')
    parser.add_argument('--conv_type',              default='ori',              type=str,       help='ori: original conv, dy: dynamic conv,')
    parser.add_argument('--test_type',              default='v1',               type=str,       help='v1: dist of the output of every layer, v2: visualize the hist of every activation map,')
    parser.add_argument('--use_wandb',              action='store_true',                        help='if use wandb to record exps')
    parser.add_argument('--pool_pos',               default='before_relu',      type=str,       help='before_relu, after_relu')
    parser.add_argument('--sub_act_mask',           action='store_true',                        help='if use sub activation mask')
    parser.add_argument('--x_thr_scale',            default=1.0,                type=float,     help='the scale of x thr')
    parser.add_argument('--pooling_type',           default='max',              type=str,       help='maxpool and avgpool')
    parser.add_argument('--weight_quantize',        default=0,                  type=int,       help='how many bit to quantize the weights')
    parser.add_argument('--qat',                    default=0,                  type=int,       help='how many bit to quantization aware training')
    parser.add_argument('--visualize',              action='store_true',                        help='visualize the attention map')
    parser.add_argument('--warmup',                 action='store_true',                        help='set lower initial learning rate to warm up the training')

    parser.add_argument('--nodes',                  default=1,                  type=int,       help='nodes')
    parser.add_argument('--rank',                   default=0,                  type=int,       help='ranking within the nodes')


    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset         = args.dataset
    batch_size      = args.batch_size
    im_size         = args.im_size
    architecture    = args.architecture
    kernel_size     = args.kernel_size
    threshold       = args.relu_threshold
    
    optimizer       = args.optimizer
    momentum        = args.momentum
    weight_decay    = args.weight_decay
    amsgrad         = args.amsgrad
    learning_rate   = args.learning_rate
    lr_reduce       = args.lr_reduce
    lr_interval_arg = args.lr_interval
    epochs          = args.epochs

    test_only       = args.test_only
    log             = args.log
    pretrained_ann  = args.pretrained_ann

    linear_dropout  = args.linear_dropout
    conv_dropout    = args.conv_dropout
    get_layer_output = args.get_layer_output
    use_init_thr    = args.use_init_thr
    get_scale       = args.get_scale
    use_x_scale     = args.use_x_scale
    net_mode        = args.net_mode
    # act_type        = args.act_type
    # thr_decay       = args.thr_decay
    hoyer_decay     = args.hoyer_decay
    hoyer_type      = args.hoyer_type
    act_mode        = args.act_mode
    start_spike_layer = args.start_spike_layer
    bn_type         = args.bn_type
    conv_type       = args.conv_type
    test_type       = args.test_type
    use_wandb       = args.use_wandb
    pool_pos        = args.pool_pos
    sub_act_mask    = args.sub_act_mask
    x_thr_scale     = args.x_thr_scale
    pooling_type    = args.pooling_type
    weight_quantize = args.weight_quantize
    qat             = args.qat
    gpu_nums        = (len(args.devices)+1) // 2

    if gpu_nums > 1:
        # distubition initialization
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print('local rank: {}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0

    values = lr_interval_arg.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*epochs))
    
    
    log_file = './logs_ann/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    
    identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')
    if test_only and pretrained_ann:
        identifier = pretrained_ann.split('/')[-1][:-4] + '_test'

    print(identifier)
    log_file+=identifier+'.log'
    
    if log:
        f= open(log_file, 'w', buffering=1)
    else:
        f=sys.stdout
    

    f.write('\n Run on time: {}'.format(datetime.datetime.now()))
            
    f.write('\n\n Arguments:')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
        
    # Training settings
    #if torch.cuda.is_available() and args.gpu:
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10
    elif dataset == 'MNIST':
        labels = 10
    elif dataset == 'IMAGENET':
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        labels = 1000
    elif dataset == 'STL10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        labels = 10
    elif dataset == 'VWW':
        labels = 2

    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        if im_size == None:
            im_size = 32
            transform_train = transforms.Compose([
                              transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize])
        else:
            transform_train = transforms.Compose([
                              transforms.Resize(im_size),
                              transforms.RandomResizedCrop(im_size),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    if dataset == 'CIFAR100':
        train_dataset   = datasets.CIFAR100(root='./cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        train_dataset   = datasets.CIFAR10(root='./cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'MNIST':
        train_dataset   = datasets.MNIST(root='./mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        test_dataset    = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'IMAGENET':
        # traindir    = os.path.join('/m2/data/imagenet', 'train')
        # valdir      = os.path.join('/m2/data/imagenet', 'val')
        traindir    = os.path.join('/nas/vista-ssd01/batl/public_datasets/ImageNet', 'train')
        valdir      = os.path.join('/nas/vista-ssd01/batl/public_datasets/ImageNet', 'val')
        if im_size == None:
            im_size = 224
            train_dataset    = datasets.ImageFolder(
                                traindir,
                                transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        else:
            train_dataset    = datasets.ImageFolder(
                                traindir,
                                transforms.Compose([
                                    transforms.Resize(im_size),
                                    transforms.RandomResizedCrop(im_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        test_dataset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(im_size+32), # 256
                                transforms.CenterCrop(im_size), # 224
                                transforms.ToTensor(),
                                normalize,
                            ]))
    elif dataset == 'STL10':
        if im_size==None:
            im_size = 96
            transform_train = transforms.Compose([
                              transforms.RandomResizedCrop(96),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize])
        else:
            transform_train = transforms.Compose([
                              transforms.Resize(im_size),
                              transforms.RandomResizedCrop(im_size),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              normalize])
        transform_test = transforms.Compose([
                         transforms.Resize(im_size),
                         transforms.CenterCrop(im_size),
                         transforms.ToTensor(),
                         normalize,
                         ])
        train_dataset = datasets.stl10.STL10(root=root+"/data/stl10_data", split="train", download=True, transform=transform_train)
        test_dataset = datasets.stl10.STL10(root=root+"/data/stl10_data", split="test", download=True, transform=transform_test)
    
    elif dataset == 'VWW':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), \
            transforms.Resize(size=(args.im_size, args.im_size)), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize(size=(args.im_size, args.im_size)), transforms.ToTensor()])

        train_dataset = VisualWakeWordsClassification_rgb(root="/home/ubuntu/data/all2014", \
            annFile="/home/ubuntu/annotations/instances_train.json", transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=8,
        #    pin_memory=True, shuffle=True)
        test_dataset = VisualWakeWordsClassification_rgb(root="/home/ubuntu/data/all2014", \
               annFile="/home/ubuntu/annotations/instances_val.json", transform=test_transform)
        #test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, num_workers=8,
        #    pin_memory=True, shuffle=False)


    if gpu_nums == 1:
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, sampler=train_sampler)
        test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=16, sampler=test_sampler)
                
    if architecture[0:3].lower() == 'vgg':
        # act_type == 'tdbn'
        model = VGG_TUNABLE_THRESHOLD_tdbn_imagenet(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size,\
            linear_dropout=linear_dropout, conv_dropout = conv_dropout, default_threshold=threshold,\
            net_mode=net_mode, hoyer_type=hoyer_type, act_mode=act_mode, bn_type=bn_type, start_spike_layer=start_spike_layer,\
            conv_type=conv_type, pool_pos=pool_pos, sub_act_mask=sub_act_mask, x_thr_scale=x_thr_scale, pooling_type=pooling_type, \
            weight_quantize=weight_quantize, im_size=im_size)


    f.write('\n{}'.format(model))
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    #model.cuda()
    #model = nn.DataParallel(model.cuda(),device_ids=[0,1,2])
    #model.cuda()
    #model = nn.DataParallel(model)
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')


    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')

        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
        # f.write('\n The threshold in ann is: {}'.format([model.threshold[key].data for key in model.threshold]))
        # if use_x_scale:
        #     model.threshold_update()
        #     f.write('\n The updated threshold in ann is: {}'.format([model.threshold[key].data for key in model.threshold]))

        #model.load_state_dict(torch.load(args.pretrained_ann, map_location='cpu')['state_dict'])

        #for param in model.features.parameters():
        #    param.require_grad = False
        #num_features = model.classifier[6].in_features
        #features = list(model.classifier.children())[:-1] # Remove last layer
        #features.extend([nn.Linear(num_features, 1000)]) # Add our layer with 4 outputs
        #model.classifier = nn.Sequential(*features) # Replace the model classifier
    
    # f.write('\n {}'.format(model)) 
    
    if torch.cuda.is_available() and args.gpu:
        model.cuda()


    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        # f.write('\npname: {}, shape: {}'.format(pname, p.shape))
        if p.ndimension() >= 2:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    if optimizer == 'SGD':
        optimizer = optim.SGD(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=learning_rate,momentum=momentum)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=learning_rate,amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
    elif optimizer == 'RMSProp':
        optimizer = optim.RMSprop(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}], 
            lr = learning_rate)
    if local_rank == 0:
        f.write('\n {}'.format(optimizer))

    if gpu_nums > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    # tensorboard_log_name = 'runs/' + identifier
    # writer = SummaryWriter(tensorboard_log_name)
    # f.write(f'tensorboead log file is saved at {tensorboard_log_name}')
    if not test_only and use_wandb and local_rank == 0:
        dir = os.path.join('wandb', identifier)
        try:
            os.mkdir(dir)
        except OSError:
            pass 
        wandb.init(project='SNN', name=identifier, group='ann', dir=dir, config=args)
        # wandb.watch(model)
    max_accuracy = 0.0


    for epoch in range(1, epochs+1):    
        start_time = datetime.datetime.now()
        if not test_only:
            train(epoch, train_loader)
        if local_rank == 0:
            test(epoch, test_loader)
        if test_only:
            break
    
    if args.visualize:
        visual_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        visualize(visual_loader, to_path=f'./visualization/{identifier}')

    f.write('\n End on time: {}'.format(datetime.datetime.now()))      
    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
