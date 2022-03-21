from Optics_simulation import Optics_simulation
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from skimage import io
from skimage.color import rgb2gray
import seaborn as sns
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.utils.data
from torch import optim
from tqdm import tqdm
from OpticalConv2d import OpticalConv2dNew
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.cuda.empty_cache()
import gc
gc.collect()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = OpticalConv2dNew(1,10,3)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = OpticalConv2dNew(10, 20, 3)
        # self.conv2 = nn.Conv2d(10,20,3, padding="same")
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(980,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train():
    best_val_loss = 10000
    best_val_acc = 0

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuarcy = 0
        val_loss = 0
        val_acc = 0
        correct = 0
        total = 0
        total_val = 0
        correct_val = 0
        for i, data in enumerate(tqdm(train_loader, desc="Epoch: " + str(epoch + 1)), 0):
            # get the inputs; data is a list of [inputs, labels]
            net.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # loss = loss.to(device)
            loss.backward(retain_graph=True)
            optimizer.step()

            # print statistics
            with torch.no_grad():
                running_loss = loss.item()
                _, predicted = torch.max(outputs.detach().cpu().data, 1)
                total += labels.size(0)
                correct += (predicted == labels.detach().cpu()).sum().item()
                running_accuarcy = 100 * correct // total
            del inputs, labels, loss, outputs
            if i%5==0:
                print(running_loss)
                print(running_accuarcy)
            torch.cuda.empty_cache()
        print(f'Epoch{epoch + 1}:      loss: {running_loss:.3f} accuaracy: {running_accuarcy}% ')
    print('Finished Training')
    return running_accuarcy, best_val_acc


if __name__=='__main__':

    print(f"running on {device}")

    # test()
    train_data = MNIST('/files/', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=0)
    test_data = MNIST('/files/', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=True, num_workers=0)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    net = Net()
    # writer.add_graph(net, images)
    # writer.flush()
    # writer.close()
    # quit()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    train()
