# %%
import logging

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from HW_1.constants import Constants
from utils.utils import Utils

Utils.configure_logging()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Device:{DEVICE}")
BATCH_SIZE = 4
logging.info(f"Batch Size:{BATCH_SIZE}")

summary_writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root=Constants.DATA_DIR, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root=Constants.DATA_DIR, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.fc_layer_input = 64 * 8 * 8

        self.fc1 = nn.Linear(self.fc_layer_input, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_layer_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model.to(DEVICE)
summary_writer.add_graph(model)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(4):  # loop over the dataset multiple times

    running_loss, running_correct, running_total = 0.0, 0, 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)

        if (i + 1) % 2000 == 0:  # print every 2000 mini-batches
            running_acc = running_correct / running_total
            print(f'[{epoch + 1}, {i + 1}]  loss:{running_loss}:.3f, acc:{running_acc}:.3f')
            summary_writer.add_scalar('Training-loss', running_loss)
            summary_writer.add_scalar('Training-acc', running_acc)
            running_loss, running_correct, running_total = 0.0, 0, 0

logging.info('Finished Training')

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

summary_writer.close()
