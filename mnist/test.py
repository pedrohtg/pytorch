from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, transform
import skimage
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable

from os import listdir
from os.path import isfile, join


PATH = './model'
IMG_PATH = './imgs/'
NUM_IMG = 6

cuda = torch.cuda.is_available()

class DigitSamples(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.len = NUM_IMG
        self.imgs = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= 0 and idx < len(self.imgs):

            imgname = join(self.root_dir, self.imgs[idx])

            img = skimage.img_as_float(io.imread(imgname, as_grey=True)).astype(np.float32)
            print(img.shape)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            #img = io.imread(join(self.root_dir, self.imgs[idx]))
            print(img.shape)
            img = img.transpose((2, 0, 1))
            print(img.shape)
            if self.transform:
                img = self.transform(img)
            print(img.shape)
            return {'image': img, 'name': self.imgs[idx]} 


dig_dataset = DigitSamples(IMG_PATH,  transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
model.load_state_dict(torch.load(PATH))
if cuda:
    model.cuda()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
test_loader = torch.utils.data.DataLoader(dig_dataset,
    batch_size=1, shuffle=True, **kwargs)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for sample in test_loader:
        data, name = sample['image'], sample['name']
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        print("Pred: " + str(pred) + "Img: " + name)

test()
