import nni
import math
import argparse
from models import *
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import argparse
import logging
#from utils import progress_bar

_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_test_acc = 0.0  # best test accuracy
best_train_acc= 0.0
start_epoch = 0
def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"], shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"], shuffle=False)
    
    if args["model"] == "vgg11":
        net = VGG("VGG11")
    if args["model"] == "vgg13":
        net = VGG("VGG13")
    if args["model"] == "vgg16":
        net = VGG("VGG16")
    if args["model"] == "vgg19":
        net = VGG("VGG19")
    if args["model"] == "googlenet":
        net = GoogLeNet()
    if args["model"] == "densenet121":
        net = DenseNet121()
    if args["model"] == "densenet161":
        net = DenseNet161()
    if args["model"] == "densenet169":
        net = DenseNet169()
    if args["model"] == "densenet201":
        net = DenseNet201()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    if args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])
def train(epoch, batches=-1):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global best_train_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        '''with torch.no_grad():
            outputs = net(inputs)'''
        loss = criterion(outputs, targets)
        #loss.requires_grad = True 企图解决之前模型过大的问题————其实最终的解决方案是减小batch size
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*(correct/total)
        #print(f"{correct} | {total}||{targets.size(0)}")
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #方便起见，就不显示图表了
        #print(loss)
        if batches > 0 and (batch_idx+1) >= batches:
            return
    if acc > best_train_acc:
        best_train_acc = acc
    return acc,best_train_acc
def test(epoch):
    global best_test_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #为了方便，就不显示图表了
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_test_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_test_acc = acc
    return acc, best_test_acc

def main_pytorch(pm):
    acc = 0.0
    best_test_acc = 0.0
    best_train_acc = 0.0
    prepare(pm)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_acc,best_train_acc = train(epoch, args.batches)
        test_acc, best_test_acc = test(epoch)
        nni.report_intermediate_result(acc)
        print(f"train acc: {train_acc}")
        print(f"test acc: {test_acc}")
    return best_train_acc,best_test_acc
if __name__ == "__main__":  
    parameter={
        "optimizer":"Adam",
        "model":"densenet201",
        "lr":0.001,
        "epochs":200,
        "batch_size":32} #单次的试验

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batches", type=int, default=-1)
    args, _ = parser.parse_known_args()

    parameter=nni.get_next_parameter()
    _logger.debug(parameter)
    train_acc,test_acc= main_pytorch(parameter)
    print(parameter)
    print(f"best train acc: {train_acc}")
    print(f"best test acc: {test_acc}")
    nni.report_final_result(end)
    print("OK")


