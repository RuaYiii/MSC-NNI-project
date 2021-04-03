import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

class ImageDataset(Dataset):
    def __init__(self, style_dir, photo_dir, size=(64,64)):
        super().__init__()
        self.style_dir = style_dir
        self.photo_dir = photo_dir
        self.style_idx = dict()
        self.photo_idx = dict()
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        for i, img in enumerate(os.listdir(self.style_dir)):
            self.style_idx[i] = img
        for i, img in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = img
    
    #定义了DataLoader中迭代采样的方式    
    def __getitem__(self, idx):
        rand_idx = int(np.random.uniform(0, len(self.style_idx.keys())))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[rand_idx])
        style_path = os.path.join(self.style_dir, self.style_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        style_img = Image.open(style_path)
        style_img = self.transform(style_img)
        return photo_img, style_img
    
    def __len__(self):
        return min(len(self.style_idx.keys()), len(self.photo_idx.keys()))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

img_ds = ImageDataset('../input/paintings/data/impressionist', '../input/paintings/data/photo_jpg')
img_dl = DataLoader(img_ds, batch_size=50, pin_memory=True, drop_last=True)
real = torch.from_numpy(np.ones((50))).long().to(device)
fake = torch.from_numpy(np.zeros((50))).long().to(device)

def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (photo, style) in enumerate(train_loader):
        photo = photo.to(device)
        style = style.to(device)
        optimizer.zero_grad()
        D_photo = model(photo)
        D_style = model(style)
        loss_photo = criterion(D_photo, fake)
        loss_style = criterion(D_style, real)
        loss = (loss_photo + loss_style) / 2
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def test(model, test_loader):
    model.eval()
    test_acc = 0
    possiblity_sum = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            possiblity_sum += F.softmax(F.sigmoid(output), dim = 1)[:, 0:1].sum().item()
            output = torch.max(output,1)[1]
            test_acc += torch.eq(output, fake).sum().item()
        return test_acc, possiblity_sum
  
Resnet = ResNet18()
Resnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Resnet.parameters(), 1e-3)

for epoch in range(40):
    loss = 0
    test_acc = 0
    possibilty_sum = 0
    loss = train(Resnet, img_dl, optimizer, criterion)
    test_acc, possibilty_sum = test(Resnet, img_dl)
    test_acc = test_acc/500
    possibilty_sum = possibilty_sum/500
    print(loss, test_acc, possibilty_sum)

torch.save(Resnet.state_dict(), 'parameter.pkl')