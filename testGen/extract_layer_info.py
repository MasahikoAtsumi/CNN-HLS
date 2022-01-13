import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import argparse
import numpy as np
from models import *
# from utils import progress_bar

def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

# ---main---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--saveastxt',    default=0,                     type=int, help='save as txt')
    parser.add_argument('--saveasnpy',    default=0,                     type=int, help='save as npy')
    parser.add_argument('--weight',       default='checkpoint/best.pth', type=str, help='root weight')
    parser.add_argument('--in_shape',     default=28,                    type=int, help='input shape')


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # ---model---
    print('==> Building model..')
    net = AlexNet()
    net = net.to(device)
    net.load_state_dict(torch.load(args.weight))
    net.eval()
    criterion = nn.CrossEntropyLoss()

    summary(net, (3,28,28)) # summary(model,(channels,H,W))

    network_info = []
    last_ch = 0
    tmp_tuple = (1,1)
    tmp_int   = 0
    in_shape = args.in_shape

    for id_loop in range(len(net.features)):
        layer_info = []
        f_list     = dir(net.features[id_loop])
        print("f_list", net.features[id_loop]._get_name() )

        if(  "Conv2d"    in net.features[id_loop]._get_name()):
            opecode = 0
        elif("MaxPool2d" in net.features[id_loop]._get_name()):
            opecode = 1
        elif("ReLU"      in net.features[id_loop]._get_name() ):
            opecode = 2

        if( "in_channels" in f_list):
            in_ch = net.features[id_loop].in_channels
        else:
            in_ch = last_ch

        if( "out_channels" in f_list):
            out_ch  = net.features[id_loop].out_channels
            last_ch = out_ch
        else:
            out_ch  = last_ch

        if( "kernel_size" in f_list):
            tmp_kernel_size = net.features[id_loop].kernel_size
            if( type(tmp_kernel_size) == type(tmp_tuple)):
                kernel_size = tmp_kernel_size[0]
            elif(type(tmp_kernel_size) == type(tmp_int)):
                kernel_size = tmp_kernel_size
        else:
            kernel_size = 0

        if( "stride" in f_list):
            tmp_stride = net.features[id_loop].stride
            if( type(tmp_stride) == type(tmp_tuple)):
                stride = tmp_stride[0]
            elif(type(tmp_stride) == type(tmp_int)):
                stride = tmp_stride
        else:
            stride = 0

        if( "padding" in f_list):
            tmp_pad = net.features[id_loop].padding
            if( type(tmp_pad) == type(tmp_tuple)):
                pad = tmp_pad[0]
            elif(type(tmp_pad) == type(tmp_int)):
                pad = tmp_pad
        else:
            pad = 0

        if opecode != 2:
            out_shape = int( (in_shape - kernel_size + 2*pad)/stride )
        else:
            out_shape = in_shape

        layer_info.append(opecode)
        layer_info.append(in_ch)
        layer_info.append(in_shape)
        layer_info.append(out_ch)
        layer_info.append(out_shape)
        layer_info.append(kernel_size)
        layer_info.append(stride)
        layer_info.append(pad)
        print(layer_info)
        print("")

        #
        network_info.append(layer_info)

        # next input shape
        in_shape = out_shape

    print(network_info)

    # ---extract parameters---
    keys = list( net.state_dict().keys() )
