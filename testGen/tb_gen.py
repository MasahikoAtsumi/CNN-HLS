import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

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

def test(object_layer, num_test):

    global best_acc
    test_loss = 0
    correct = 0
    total = 0

    def_txtpath = '.\\iotxt\\'
    create_dir(def_txtpath)
    def_npypath = '.\\ionpy\\'
    create_dir(def_txtpath)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            if batch_idx == num_test:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _data  = net.features(inputs)
            o_data = _data.detach().numpy()
            i_data = inputs.detach().numpy()

            if args.saveasnpy == 1:
                save_npypath = def_npypath + '\\' + str(batch_idx) + '\\'
                create_dir(save_npypath)
                np.save(save_npypath + '\\out_' + str(batch_idx), o_data)
                np.save(save_npypath + '\\in_'  + str(batch_idx),  i_data )

            if args.saveastxt == 1:
                save_txtpath = def_txtpath + '\\' + str(batch_idx) + '\\'
                create_dir(save_txtpath)
                np.savetxt(save_txtpath + '\\out_'  + str(batch_idx) + '.txt', o_data.flatten(), delimiter=',', fmt ='%.5f')
                np.savetxt(save_txtpath + '\\in_'   + str(batch_idx) + '.txt', i_data.flatten(), delimiter=',', fmt ='%.5f')

            for layer_id, p_module in enumerate(net.modules()):
                if layer_id < 3:
                    y = p_module(inputs)
                    bench_output = y.detach().numpy()
                    bench_output = bench_output.reshape(-1,)
                    next_input   = y
                else:
                    y = p_module(next_input)
                    bench_output = y.detach().numpy()
                    bench_output = bench_output.reshape(-1,)
                    next_input   = y


                if layer_id == object_layer - 1:
                    if batch_idx == 0:
                        print(p_module)
                        print('output.dimension',y.shape)
                        print('output', y.shape)
                    fname = 'output_targetLayer.txt'
                    if args.saveastxt == 1:
                        np.savetxt(save_txtpath + '\\in_target_'  + str(batch_idx) + '.txt', bench_output, fmt="%.5f", delimiter=",")

                if layer_id == object_layer:
                    if batch_idx == 0:
                        print(p_module)
                        print('output.dimension',y.shape)
                        print('output', y.shape)
                    fname = 'output_targetLayer.txt'
                    if args.saveastxt == 1:
                        np.savetxt(save_txtpath + '\\out_target_'  + str(batch_idx) + '.txt', bench_output, fmt="%.5f", delimiter=",")
                    break

            '''
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            '''

# ---main---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--saveastxt',    default=1,                     type=int, help='save as txt')
    parser.add_argument('--saveasnpy',    default=1,                     type=int, help='save as npy')
    parser.add_argument('--weight',       default='checkpoint/best.pth', type=str, help='root weight')
    parser.add_argument('--target_layer', default=0,                     type=int, help='target layer')
    parser.add_argument('--howmany',      default=1,                   type=int, help='target layer')

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

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

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

    # ---extract features---
    test(args.target_layer, args.howmany)

    # ---extract parameters---
    keys = list( net.state_dict().keys() )
    k_id = 0
    for param in net.parameters():
        save_data = param.detach().numpy()
        save_name = keys[k_id]
        k_id += 1
        if args.saveastxt == 1:
            create_dir('.\\txt\\')
            np.savetxt('.\\txt\\' + str(save_name)+'.txt', save_data.flatten(), delimiter=',', fmt ='%.5f')
        if args.saveasnpy == 1:
            create_dir('.\\npy\\')
            np.save(   '.\\npy\\' + str(save_name), save_data)
