from __future__ import print_function
import argparse
from asyncore import write
from fileinput import filename
from tokenize import Name
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from models.vgg_tunable_spiking import *
from models.my_prune import *
import sys
import os
import shutil
import argparse

import wandb


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

def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):
    
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    model.network_update(timesteps=timesteps, leak=1.0)
    pos=0
    thresholds=[]
    
    def find(layer):
        max_act=0
        
        f.write('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    f.write(' {}'.format(thresholds))
                    model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
    
    if architecture.lower().startswith('vgg'):              
        for l in model.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        for c in model.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(c[0]) == len(model.classifier) -1):
                    break
                else:
                    find(int(l[0])+int(c[0])+1)

    if architecture.lower().startswith('res'):
        for l in model.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                find(int(l[0]))
        
        pos = len(model.pre_process)

        for i in range(1,5):
            layer = model.layers[i]
            for index in range(len(layer)):
                for l in range(len(layer[index].residual)):
                    if isinstance(layer[index].residual[l],nn.Conv2d):
                        pos = pos +1

        for c in model.classifier.named_children():
            if isinstance(c[1],nn.Linear):
                if (int(c[0])==len(model.classifier)-1):
                    break
                else:
                    find(int(c[0])+pos)

    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    global learning_rate
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    model.train()
    local_time = datetime.datetime.now()  
    total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    for batch_idx, (data, target) in enumerate(train_loader):
               
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, act_out, relu_batch_num = model(data, epoch=epoch)
        loss = F.cross_entropy(output,target)
        data_size = data.size(0)
        loss.backward()
        optimizer.step()    

        losses.update(loss.item(), data_size)
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        top1.update(correct.item()/data_size, data_size) 
  
        if (batch_idx+1) % train_acc_batches == 0 or (batch_idx < 5 and epoch == 1):
            temp1 = []
            temp2 = []
            total_batch_num = relu_batch_num
            for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[round(value.item(),5)]
            # for l in relu_total_num.keys():
            #     # f.write('\tIn layer {}, the rate of ReLU is {:.4f}%'.format(l, relu_total_num[l][0]/relu_total_num[l][1]*100.0))
            #     total_num += relu_total_num[l]
            f.write('\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}% timesteps: {}, time: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        total_batch_num[0]/total_batch_num[-1]*100.0,
                        total_batch_num[1]/total_batch_num[-1]*100.0,
                        total_batch_num[2]/total_batch_num[-1]*100.0,
                        model.timesteps,
                        datetime.timedelta(seconds=(datetime.datetime.now() - local_time).seconds)
                        )
                    )
            f.write('\nthresold: {}, leak: {}'.format(temp1, temp2))
            local_time = datetime.datetime.now()
        total_num += relu_batch_num
    wandb.log({'loss': losses.avg,}, step=epoch)
    wandb.log({'training_acc': top1.avg}, step=epoch)
    wandb.log({'Relu_less_eq_0': total_num[0]/total_num[-1]*100}, step=epoch)
    wandb.log({'Relu_between_0_thr': total_num[1]/total_num[-1]*100}, step=epoch)
    wandb.log({'Relu_laeger_eq_thr': total_num[2]/total_num[-1]*100}, step=epoch)
    
    # for l in relu_total_num.keys():
    #     total_num += relu_total_num[l]

    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%, time: {}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        total_num[0]/total_num[-1]*100.0,
                        total_num[1]/total_num[-1]*100.0,
                        total_num[2]/total_num[-1]*100.0,
                        datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
                        )
                    )
 
def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    
    if args.test_only:
        temp1 = []  
        temp2 = []  
        for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):    
            temp1 = temp1+[round(value.item(),2)]   
        for key, value in sorted(model.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))): 
            temp2 = temp2+[round(value.item(),2)]   
        f.write('\n Thresholds: {}, leak: {}'.format(temp1, temp2))

    with torch.no_grad():
        model.eval()
        global max_accuracy

        if args.layer_output:
            total_output = {}
            for batch_ids, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available() and args.gpu:
                    data, target = data.cuda(), target.cuda()
                output, spike_count, layer_output = model(data)
                total_output[batch_ids] = layer_output.copy()
            
            torch.save(total_output, 'classifier_layer_output')

        relu_total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            epoch = False if args.test_only else epoch
            output, act_out, relu_batch_num = model(data, epoch=epoch) 

            if args.test_only and batch_idx==0:
                res = {}
                total_net_output = torch.tensor([]).cpu()
                for l in act_out.keys():
                    # act_reg += (torch.sum(torch.abs(act_out[l]))**2 / torch.sum((act_out[l])**2))
                    total_net_output = torch.cat((total_net_output, act_out[l].view(-1).cpu()))
                    if batch_idx == 0:
                        f.write(f'\nlayer {l} shape: {act_out[l].shape}, net_output: {total_net_output.shape}')
                        res[l] =  act_out[l].view(-1).cpu().numpy()
                        # writer.add_histogram(f'Dist/layer {l} distribution', act_out[l].view(-1).cpu().numpy())
                for l in relu_batch_num.keys():
                    if batch_idx==0:
                        f.write('\nIn layer {}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%'.format(
                        l, 
                        relu_batch_num[l][0]/relu_batch_num[l][-1]*100.0,
                        relu_batch_num[l][1]/relu_batch_num[l][-1]*100.0,
                        relu_batch_num[l][2]/relu_batch_num[l][-1]*100.0))
                    relu_total_num += relu_batch_num[l]

                res['total'] = total_net_output.view(-1).cpu().numpy()
                # writer.add_histogram('Dist/output distribution', total_net_output.view(-1).cpu().numpy())
                torch.save(res, 'network_output/'+identifier)
            if not args.test_only:
                relu_total_num += relu_batch_num
            
            #for key in spike_count.keys():
            #    print('Key: {}, Average: {:.3f}'.format(key, (spike_count[key].sum()/spike_count[key].numel())))
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                f.write('\n Images {}/{} Accuracy: {}/{}({:.4f})'
                    .format(
                    test_loader.batch_size*(batch_idx+1),
                    len(test_loader.dataset),
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
        
        temp1 = []
        temp2 = []
        for key, value in sorted(model.threshold.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp1 = temp1+[value.item()]
        for key, value in sorted(model.leak.items(), key=lambda x: (int(x[0][1:]), (x[1]))):
                temp2 = temp2+[value.item()]
        
        # f.write('\n')
        # total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        # for l in relu_total_num.keys():
        #     # f.write('\tIn layer {}, the rate of ReLU is {:.4f}%'.format(l, relu_total_num[l][0]/relu_total_num[l][1]*100.0))
        #     total_num += relu_total_num[l]

        #if epoch>5 and top1.avg<0.15:
        #    f.write('\n Quitting as the training is not progressing')
        #    exit(0)


        if top1.avg >= max_accuracy and top1.avg > 0.80:
            max_accuracy = top1.avg
        # if top1.avg>max_accuracy or (prune_epoch !=0 and epoch%prune_epoch == 0):
            # max_accuracy = top1.avg if top1.avg>max_accuracy else max_accuracy
             
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : temp2,
                    'activation'            : activation
                }
            # try:
            #     os.mkdir('./trained_snn_models/'+identifier+'/')
            # except OSError:
            #     pass 
            # filename = './trained_snn_models/'+identifier+ '/'+ 'epoch_' + str(epoch) + '_' + str(max_accuracy) +'.pth'
            filename = './trained_snn_models/'+identifier+ '.pth'
            if not args.dont_save and not args.test_only:
                torch.save(state,filename)

            if prune_epoch != 0 and epoch == args.epochs:
                for index, module in model.features.named_children():
                    if isinstance(module, nn.Conv2d):
                        prune.remove(module, 'weight')
                for index, module in model.classifier.named_children():
                    if isinstance(module, nn.Linear) and int(index) < 2:
                        prune.remove(module, 'weight')
                pruned_file = './trained_snn_models/'+identifier+ '/'+ 'pruned_epoch_' + str(epoch) + str(max_accuracy) +'.pth'
                state['state_dict'] = model.state_dict()
                torch.save(state, pruned_file)
        
        # print(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds))
        if not args.test_only:
            wandb.log({'test_acc': top1.avg}, step=epoch)
        f.write('\ntest_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%, time: {}\n'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            relu_total_num[0]/relu_total_num[-1]*100,
            relu_total_num[1]/relu_total_num[-1]*100,
            relu_total_num[2]/relu_total_num[-1]*100,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--description',            default='exp desc',         type=str,       help='description for the exp')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100','IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG4','VGG6','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=30,                 type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=20,                 type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.3,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear'])
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.95,               type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--betas',                  default='0.9,0.999',        type=str,       help='betas for Adam optimizer'  )
    parser.add_argument('--dropout',                default=0.5,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=1000,               type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--resume',                 default='',                 type=str,       help='resume training from this state')
    parser.add_argument('--dont_save',              action='store_true',                        help='don\'t save training model during testing')
    parser.add_argument('--prune_epoch',            default='0',                                help='prune the neural network evry k epoch')
    parser.add_argument('--layer_output',           action='store_true',                        help='get the output before relu&dropout for every layer')
    parser.add_argument('--use_init_thr',           action='store_true',                        help='use the inital threshold')
    parser.add_argument('--decay',                  default=0.001,              type=float,     help='weight decay for regularizer (default: 0.001)')
    parser.add_argument('--reg_type',               default=2,                  type=int,       metavar='R',help='regularization type: 0:None 1:L1 2:Hoyer 3:HS')
    parser.add_argument('--act_decay',              default=-1.0,               type=float,     help='weight decay for activation function regularizer (default: -1)')


    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    beta1               = float(args.betas.split(',')[0])
    beta2               = float(args.betas.split(',')[1])
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches
    resume              = args.resume
    start_epoch         = 1
    max_accuracy        = 0.0
    prune_epoch         = int(args.prune_epoch)
    
    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    # identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'final'
    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M')
    if args.test_only:
        identifier = pretrained_snn.split('/')[-1][:-4] + '_test'
    print(identifier)
    log_file+=identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file

    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
    elif dataset == 'IMAGENET':
        normalize   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    #normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = './cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = './cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
    
    elif dataset == 'MNIST':
        trainset   = datasets.MNIST(root='./mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        testset    = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

    elif dataset == 'IMAGENET':
        labels      = 1000
        traindir    = os.path.join('/local/a/imagenet/imagenet2012/', 'train')
        valdir      = os.path.join('/local/a/imagenet/imagenet2012/', 'val')
        trainset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])) 

    train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, dropout=dropout, kernel_size=kernel_size, scaling_factor = scaling_factor, dataset=dataset)
    
    elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN(resnet_name = architecture, activation = activation, labels=labels, timesteps=timesteps,leak=leak, default_threshold=default_threshold, dropout=dropout, dataset=dataset)
      
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    #model = nn.DataParallel(model) 
    #model = nn.parallel.DistributedDataParallel(model)
    #pdb.set_trace()

    if pretrained_ann:
        
        
      
        state = torch.load(pretrained_ann, map_location='cpu')
        
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        #If thresholds present in loaded ANN file
        #if 'threshold' in state['state_dict'].keys():
        #    thresholds = state['thresholds']
        #    f.write('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
        #    model.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        #else:
            #thresholds = find_threshold(batch_size=512, timesteps=500, architecture=architecture)
        #thresholds = [0.8727,1.4557,0.9344,0.5781,0.5672,0.2121,0.1086,0.0641,0.0747,0.1266,0.1786,0.3390,0.6676,0.6794,0.5203]
        # f.write('\n The threshold in ann is: {}'.format([thr for thr in model.threshold.items()]))
        model.threshold_update(scaling_factor = scaling_factor)
        # f.write('\n After update, the threshold is: {}'.format([thr for thr in model.threshold.items()]))
            
            #Save the threhsolds in the ANN file
        #temp = {}
        #for key,value in state.items():
        #    temp[key] = value
            #temp['thresholds'] = thresholds
        #temp['thresholds'] = thresholds
            
        #torch.save(temp, pretrained_ann)
    
    elif pretrained_snn:
                
        state = torch.load(pretrained_snn, map_location='cpu')
        f.write('\n The threshold in snn is: {}'.format([thr for thr in state['thresholds']]))
        if args.use_init_thr:
            init_model = torch.load('trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth', map_location='cpu')
            init_state = init_model['state_dict']
            for key in init_state.keys():
                if key[:9] == 'threshold':
                    state['state_dict'][key] = init_state[key]

        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        f.write('\n Info: Accuracy of loaded snn model: {}'.format(state['accuracy']))
        # f.write('\n The threshold in snn is: {}'.format([thr for thr in state['thresholds']]))
        if args.use_init_thr:
            model.threshold_update(scaling_factor = scaling_factor)

       
    # f.write('\n {}'.format(model))
    
    #model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

       
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)
    
    f.write('\n {}'.format(optimizer))
        
    # find_threshold() alters the timesteps and leak, restoring it here
    model.network_update(timesteps=timesteps, leak=leak)
    
    if resume:
        f.write('\n Resuming from checkpoint {}'.format(resume))
        state = torch.load(resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
              
        epoch           = state['epoch']
        start_epoch     = epoch + 1
        max_accuracy    = state['accuracy']
        optimizer.load_state_dict(state['optimizer'])
        for param_group in optimizer.param_groups:
            learning_rate =  param_group['lr']

        f.write('\n Loaded from resume epoch: {}, accuracy: {:.4f} lr: {:.1e}'.format(epoch, max_accuracy, learning_rate))
    # model = nn.DataParallel(model)
    if not args.test_only:
        dir = os.path.join('wandb', identifier)
        try:
            os.mkdir(dir)
        except OSError:
            pass 
        wandb.init(project='SNN', name=identifier, group='ann', dir=dir, config=args)
        # wandb.watch(model)
    for epoch in range(start_epoch, epochs+1):
        start_time = datetime.datetime.now()
        
        if not args.test_only:
            train(epoch)
        if prune_epoch != 0 and epoch%prune_epoch == 1:
            gama = (epoch-1) // prune_epoch * 0.1 + 0.1
            f.write('\n Before pruning: -----------------\n')
            test(epoch)
            scale_x_ = model.scale_factor.clone().detach()
            t_index = s_index = 0
            pruned_weight_num = 0.0
            total_weight_num = 0.0
            for index, module in model.features.named_children():
                if isinstance(module, nn.Conv2d):
                    pruned_num, total_num = check_sparsity(f, f'conv{index}', module, getattr(model.threshold, 't'+str(t_index)).clone().detach(), scale_x_[s_index], gama)
                    s_index += 1
                    pruned_weight_num += pruned_num
                    total_weight_num += total_num
                t_index += 1
            for index, module in model.classifier.named_children():
                if isinstance(module, nn.Linear) and int(index) < 2:
                    pruned_num, total_num = check_sparsity(f, f'fc{index}', module, getattr(model.threshold, 't'+str(t_index)).clone().detach(), scale_x_[s_index], gama) 
                    s_index += 1
                    pruned_weight_num += pruned_num
                    total_weight_num += total_num
                t_index += 1
            f.write('\n After pruning: -----------------\n')
            f.write('\nThe total sparsity is {:.2f}%\n'.format(100. * pruned_weight_num / total_weight_num))
        test(epoch)
    f.write('\n End on time: {}'.format(datetime.datetime.now()))
    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))



