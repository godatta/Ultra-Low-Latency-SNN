import argparse
from operator import ge

from models.vgg_tunable_threshold_tdbn import VGG_TUNABLE_THRESHOLD_tdbn
from models.resnet_tunable_threshold import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
import numpy as np
import json
import pickle
from utils.net_utils import *
# from torch.utils.tensorboard import SummaryWriter
import wandb

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
    thr_losses = AverageMeter('Loss')
    act_losses = AverageMeter('Loss')
    total_losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']

    #if epoch in lr_interval:
    #else:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = param_group['lr'] / ((1000-2*(epoch-1))/(998-2*(epoch-1)))
    #        learning_rate = param_group['lr']
    
    #total_correct   = 0
    relu_total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
    test_hoyer_thr = torch.tensor([0.0]*15)
    model.train() # this is impoetant, cannot remove
    for batch_idx, (data, target) in enumerate(loader):
        
        #start_time = datetime.datetime.now()

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
                
        optimizer.zero_grad()
        #output, _ = model(data)
        # if act_type == 'relu':
        #     output, model_thr, relu_batch_num, act_out, thr_out = model(data, epoch)
        # else:
        output, model_thr, relu_batch_num, act_out = model(data, epoch)
        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()

        data_size = data.size(0)
        act_loss = hoyer_decay*act_out
        # total_loss = loss + act_loss
        total_loss = loss + act_loss
        # if act_type == 'relu':
        #     thr_loss = thr_decay*thr_out
        #     total_loss += thr_loss
        total_loss.backward(inputs = list(model.parameters()))
        
        optimizer.step()       
        
        losses.update(loss.item(),data_size)
        act_losses.update(act_loss, data_size)
        # if act_type == 'relu':
        #     thr_losses.update(thr_loss, data_size)
        # reg_losses.update(reg_loss, data_size)
        total_losses.update(total_loss.item(), data_size)

        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        top1.update(correct.item()/data_size, data_size)
        relu_total_num += relu_batch_num
        test_hoyer_thr += model.test_hoyer_thr
        # torch.cuda.empty_cache()
        if epoch == 1 and batch_idx < 5:
            f.write('\nbatch: {}, train_loss: {:.4f}, act_loss: {:.4f}, thr_loss: {:.4f} total_train_loss: {:.4f} '.format(
            batch_idx,
            losses.avg,
            act_losses.avg,
            thr_losses.avg,
            total_losses.avg,
            ))
            f.write('train_acc: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%, time: {}'.format(
            top1.avg,
            relu_total_num[0]/relu_total_num[-1]*100,
            relu_total_num[1]/relu_total_num[-1]*100,
            relu_total_num[2]/relu_total_num[-1]*100,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            ))
        # if batch_idx ==6:
        #     exit()
    # writer.add_scalars('Loss/train', {
    #     'loss': losses.avg,
    #     # 'loss_reg': reg_loss,
    #     'loss_act': act_losses.avg,
    #     'total_loss': total_losses.avg,
    #     }, epoch)
    # writer.add_scalar('Accuracy/train', top1.avg, epoch)
    # writer.add_scalar('Relu/less_eq_0', relu_total_num[0]/relu_total_num[-1]*100, epoch)
    # writer.add_scalar('Relu/between_0_thr', relu_total_num[1]/relu_total_num[-1]*100, epoch)
    # writer.add_scalar('Relu/laeger_eq_thr', relu_total_num[2]/relu_total_num[-1]*100, epoch)
    if use_wandb:
        wandb.log({
            'loss': losses.avg,
            'loss_act': act_losses.avg,
            'loss_thr': thr_losses.avg,
            'total_loss': total_losses.avg
        }, step=epoch)
        for i in range(len(model.test_hoyer_thr)):
            wandb.log({f'hoyer_thr_{i}': test_hoyer_thr[i]/batch_idx}, step=epoch)
        wandb.log({'training_acc': top1.avg}, step=epoch)
        wandb.log({'Relu_less_eq_0': relu_total_num[0]/relu_total_num[-1]*100}, step=epoch)
        wandb.log({'Relu_between_0_thr': relu_total_num[1]/relu_total_num[-1]*100}, step=epoch)
        wandb.log({'Relu_laeger_eq_thr': relu_total_num[2]/relu_total_num[-1]*100}, step=epoch)
    f.write('\n The threshold in ann is: {}'.format([p.data for p in model_thr]))
    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, act_loss: {:.4f}, thr_loss: {:.4f} total_train_loss: {:.4f} '.format(
            epoch,
            learning_rate,
            losses.avg,
            act_losses.avg,
            thr_losses.avg,
            total_losses.avg,
            )
        )
    f.write('train_acc: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%, time: {}'.format(
            top1.avg,
            relu_total_num[0]/relu_total_num[-1]*100,
            relu_total_num[1]/relu_total_num[-1]*100,
            relu_total_num[2]/relu_total_num[-1]*100,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

def test(epoch, loader):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')
    act_losses = AverageMeter('Loss') 
    thr_losses = AverageMeter('Loss')
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
            
        relu_total_num = torch.tensor([0.0, 0.0, 0.0, 0.0])
        test_hoyer_thr = torch.tensor([0.0]*15)
        for batch_idx, (data, target) in enumerate(loader):
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            # if get_scale and test_only:
            #     output, thresholds, relu_batch_num, act_out = model(data, -2)
            if test_only and get_layer_output and batch_idx < 100 and epoch == 1:
                output, thresholds, relu_batch_num, act_out = model(data, -1)
                for l in act_out.keys():
                    act_out[l] = act_out[l][act_out[l]>0]
                    act_out[l] = act_out[l][act_out[l]<1.0]
                    if l not in total_output:
                        total_output[l] = torch.tensor([])
                    total_output[l] = torch.cat((total_output[l], act_out[l].cpu()))

            elif test_only and batch_idx <= 0 and epoch == 1:
                output, thresholds, relu_batch_num, act_out = model(data, -1)
                # act_reg = 0.0
                res = {}
                total_net_output = torch.tensor([])
                for l in act_out.keys():
                    # act_reg += (torch.sum(torch.abs(act_out[l]))**2 / torch.sum((act_out[l])**2))
                    if test_type == 'v2' and len(act_out[l].shape) == 4:
                        N,C,W,H = act_out[l].shape
                        # if overflow, output heatmap
                        # if W >= 8:
                        print('the max of {}, is {}'.format(l, torch.max(act_out[l])))
                        # act_out[l][act_out[l] < 1.0] = 0.0
                        res[l] = torch.sum(act_out[l], dim=(0,1)).cpu()
                        # res[l] = res[l][2]
                        plot_output[l] = plot_output.get(l, 0.0) + res[l]
                        # if not overflow, output histgram
                        # else:
                        #     index_M = 2**(torch.arange(0, W*H).reshape(W,H).cuda())
                        #     act_out[l][act_out[l]>0] = 1.0
                        #     # print('act_out sum: {}, nums: {}'.format(torch.sum(act_out[l]), torch.count_nonzero(act_out[l])))
                        #     act_out[l] = act_out[l]*index_M
                        #     res[l] = torch.sum(act_out[l], dim=(2,3))
                        #     print(f'before res: {res[l].shape}')
                        #     res[l] = res[l][:,0].view(-1).cpu()
                        #     print(f'after res: {res[l].shape}')
                        #     plot_output[l] = torch.cat((plot_output.get(l, torch.tensor([])), res[l]))
                    elif test_type == 'v2' and len(act_out[l].shape) == 2:
                        N,D = act_out[l].shape
                        res[l] = torch.sum(act_out[l], dim=(0)).cpu()
                        plot_output[l] = plot_output.get(l, 0.0) + res[l]
                        # act_out[l][act_out[l]>0] = 1
                        # index_M = (torch.arange(1,D+1.).cuda())
                        # act_out[l] = act_out[l]*index_M
                        # print(torch.sum(index_M))
                        # print(torch.sum(act_out[l], dim=1))
                        # res[l] = torch.sum(act_out[l], dim=1).cpu()
                        # res[l] = (act_out[l]@index_M).cpu()
                        # plot_output[l] = torch.cat((plot_output.get(l, torch.tensor([])), res[l]))
                    else:
                        total_net_output = torch.cat((total_net_output, act_out[l].view(-1).cpu()))
                        res[l] =  act_out[l].view(-1).cpu().numpy()
                                          
                    # writer.add_histogram(f'Dist/layer {l} distribution', act_out[l].view(-1).cpu().numpy())
                    if batch_idx == 1:
                        f.write(f'\nlayer {l} shape: {act_out[l].shape}, output shape: {res[l].shape}')
                        if test_type == 'v2':
                            f.write(f', plot out shape: {plot_output[l].shape}')
                res['total'] = total_net_output.view(-1).cpu().numpy()
                # writer.add_histogram('Dist/output distribution', total_net_output.view(-1).cpu().numpy())
                torch.save(res, 'network_output/'+identifier)

            else:
                # if act_type == 'relu':
                #     output, thresholds, relu_batch_num, act_out, thr_out = model(data, epoch)
                # else:
                output, thresholds, relu_batch_num, act_out = model(data, epoch)

            #output, thresholds = model(data)
            #dis.extend(act)
            loss = F.cross_entropy(output,target)
            data_size = data.size(0)
            if get_scale:
                if len(hoyer_thr_per_batch) <= 0:
                    hoyer_thr_per_batch = (model.test_hoyer_thr).cpu().numpy()
                else:
                    hoyer_thr_per_batch = np.vstack((hoyer_thr_per_batch, (model.test_hoyer_thr).cpu().numpy()))
                act_loss = 0.0
                # print('hoyer_thr_per_batch shape: {}'.format(hoyer_thr_per_batch.shape))
            elif test_only:
                act_loss = 0.0
            else:
                act_loss = hoyer_decay*act_out
            total_loss = loss+act_loss
            # if act_type == 'relu':
            #     thr_loss = thr_decay*thr_out
            #     total_loss += thr_loss
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            act_losses.update(act_loss, data_size)
            losses.update(loss.item(), data_size)
            # if act_type == 'relu':
            #     thr_losses.update(thr_loss, data_size)
            total_losses.update(total_loss.item(), data_size)
            top1.update(correct.item()/data_size, data_size)

            relu_total_num += relu_batch_num
            test_hoyer_thr += model.test_hoyer_thr
        #with open('percentiles_resnet20_cifar100.json','w') as f:
        #    json.dump(percentiles, f)

        #with open('thresholds_resnet20_cifar100_new', 'wb') as fp:
        #    pickle.dump(thresholds, fp)
        
        #with open('activations','wb') as f:
        #    pickle.dump(dis, f)

        #if epoch>30 and top1.avg<0.15:
        #    f.write('\n Quitting as the training is not progressing')
        #    exit(0)
        final_avg = np.array([(p.data)/(batch_idx+1) for p in test_hoyer_thr])
        if test_only and test_type == 'v2':
            torch.save(plot_output, 'network_output/'+identifier+'_v2')
        if get_scale:
            torch.save(hoyer_thr_per_batch, 'output/my_hoyer_x_scale_factor')
            torch.save(final_avg, 'output/my_hoyer_x_scale_factor_final_avg')
        if get_layer_output:
            torch.save(total_output, 'output/ann_tdbn_layer_output')
        if not test_only and use_wandb:
            wandb.log({'test_acc': top1.avg}, step=epoch)
        # writer.add_scalar('Accuracy/test', top1.avg, epoch)
        if (top1.avg>=max_accuracy) and top1.avg>0.88:
            max_accuracy = top1.avg
            # if not test_only:
            #     wandb.run.summary["best_accuracy"] = top1.avg
            state = {
                    'accuracy'      : max_accuracy,
                    'epoch'         : epoch,
                    'state_dict'    : model.state_dict(),
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
        f.write('\nEpoch: {}, best: {:.4f}, test_loss: {:.4f}, act_loss: {:.4f}, thr_loss: {:.4f}, total_test_loss: {:.4f}, '.format(
            epoch,
            max_accuracy,
            losses.avg,
            act_losses.avg,
            thr_losses.avg,
            total_losses.avg,
            )
        )
        f.write('test_acc: {:.4f}, output 0: {:.2f}%, relu: {:.2f}%, output threshold: {:.2f}%, time: {}\n'.format(
            top1.avg,
            relu_total_num[0]/relu_total_num[-1]*100,
            relu_total_num[1]/relu_total_num[-1]*100,
            relu_total_num[2]/relu_total_num[-1]*100,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

        f.write('\nThe hoyer thr in ann is: {}'.format([(p.data)/(batch_idx+1) for p in test_hoyer_thr]))

        # f.write('\n Time: {}'.format(
        #     datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
        #     )
        # )

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

    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100', 'IMAGENET'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')


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
    parser.add_argument('--act_mode',               default='v1',               type=str,       help='fixed,mean,sum,channelwise(cw), spike: the type of activation function')
    parser.add_argument('--start_spike_layer',      default=20,                 type=int,       help='start_spike_layer')
    parser.add_argument('--bn_type',                default='bn',               type=str,       help='bn: , tdbn: , fake: the type of batch normalization')
    parser.add_argument('--conv_type',              default='ori',              type=str,       help='ori: original conv, dy: dynamic conv,')
    parser.add_argument('--test_type',              default='v1',               type=str,       help='v1: dist of the output of every layer, v2: visualize the hist of every activation map,')
    parser.add_argument('--use_wandb',              action='store_true',                        help='if use wandb to record exps')
    parser.add_argument('--pool_pos',               default='before_relu',      type=str,       help='before_relu, after_relu')
    parser.add_argument('--sub_act_mask',           action='store_true',                        help='if use sub activation mask')
    parser.add_argument('--x_thr_scale',            default=1.0,                type=float,     help='the scale of x thr')
    parser.add_argument('--pooling_type',           default='max',              type=str,       help='maxpool and avgpool')



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

    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
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
        traindir    = os.path.join('/m2/data/imagenet', 'train')
        valdir      = os.path.join('/m2/data/imagenet', 'val')
        train_dataset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        test_dataset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))


    
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    if architecture[0:3].lower() == 'vgg':
        # act_type == 'tdbn'
        model = VGG_TUNABLE_THRESHOLD_tdbn(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size,\
            linear_dropout=linear_dropout, conv_dropout = conv_dropout, default_threshold=threshold,\
            net_mode=net_mode, hoyer_type=hoyer_type, act_mode=act_mode, bn_type=bn_type, start_spike_layer=start_spike_layer,\
            conv_type=conv_type, pool_pos=pool_pos, sub_act_mask=sub_act_mask, x_thr_scale=x_thr_scale, pooling_type=pooling_type)

    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet12':
            model = ResNet12(labels=labels, dropout=dropout, default_threshold=threshold)
        elif architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout, default_threshold=threshold)
        elif architecture.lower() == 'resnet34':
            model = ResNet34(labels=labels, dropout=dropout, default_threshold=threshold) 
    f.write('\n{}'.format(model))
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    #model.cuda()
    #model = nn.DataParallel(model.cuda(),device_ids=[0,1,2])
    #model.cuda()
    #model = nn.DataParallel(model)

    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        
        if use_init_thr:
            init_thr = []
            init_model = torch.load('trained_models_ann/ann_vgg16_cifar10_4.0_0.2lr_decay.pth', map_location='cpu')
            init_state = init_model['state_dict']
            f.write('\noriginal threshold: ')
            for key in init_state.keys():
                if key[:9] == 'threshold':
                    f.write('{:.4f}, '.format(state['state_dict'][key]))
                    state['state_dict'][key] = init_state[key]

        missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
        f.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))        
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
        # f.write('\n The threshold in ann is: {}'.format([model.threshold[key].data for key in model.threshold]))
        # if use_x_scale:
        #     model.threshold_update()
        #     f.write('\n The updated threshold in ann is: {}'.format([model.threshold[key].data for key in model.threshold]))
        '''
        state=torch.load(args.pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state[key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        
        #model.load_state_dict(cur_dict)
        '''
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
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
    
    f.write('\n {}'.format(optimizer))

    # tensorboard_log_name = 'runs/' + identifier
    # writer = SummaryWriter(tensorboard_log_name)
    # f.write(f'tensorboead log file is saved at {tensorboard_log_name}')
    if not test_only and use_wandb:
        dir = os.path.join('wandb', identifier)
        try:
            os.mkdir(dir)
        except OSError:
            pass 
        wandb.init(project='SNN', name=identifier, group='ann', dir=dir, config=args)
        # wandb.watch(model)
    max_accuracy = 0.0
    #compute_mac(model, dataset)
    # model = nn.DataParallel(model)
    # model.train()
    # # try to freeze model's weights
    # parser_args = {'learn_batchnorm': False,'bn_bias_only': False, 'tune_batchnorm': False}
    if sub_act_mask:
        freeze_model_weights(model=model)
    for epoch in range(1, epochs+1):    
        start_time = datetime.datetime.now()
        if not test_only:
            train(epoch, train_loader)
        test(epoch, test_loader)
        if test_only:
            break
    
    f.write('\n End on time: {}'.format(datetime.datetime.now()))      
    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))

