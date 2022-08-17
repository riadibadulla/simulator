import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/")
torch.set_num_threads(16)
torch.cuda.empty_cache()
import gc
import sys
gc.collect()
torch.manual_seed(2022)
import getopt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
from matplotlib import pyplot as plt
from pylab import savefig
import torch.optim as optim
import pandas as pd

args = sys.argv[1:]
optlist, args = getopt.getopt(args,'',["batchsize=","model=","epochs="])
optlist = dict(optlist)
print(optlist["--model"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on {device}")
print(torch.cuda.device_count())
from torchvision.transforms.transforms import RandomRotation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

batch_size = int(optlist["--batchsize"])

trainset = torchvision.datasets.CIFAR100(root='files/', train=True,
                                        download=True, transform=transform)

train_set, val_set = torch.utils.data.random_split(trainset, [40000, 10000])

validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='files/', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


classes = ('apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',' bowl',' boy',' bridge',' bus',' butterfly',' camel',
          ' can',' castle',' caterpillar',' cattle',' chair',' chimpanzee',' clock',' cloud',' cockroach',' couch',' cra',' crocodile',' cup',' dinosaur',' dolphin',
          ' elephant',' flatfish',' forest',' fox',' girl',' hamster',' house',' kangaroo',' keyboard',' lamp',' lawn_mower',' leopard',' lion',' lizard',' lobster',' man',
          ' maple_tree',' motorcycle',' mountain',' mouse',' mushroom',' oak_tree',' orange',' orchid',' otter',' palm_tree',' pear',' pickup_truck',' pine_tree',' plain',' plate',
          ' poppy',' porcupine',' possum',' rabbit',' raccoon',' ray',' road',' rocket',' rose',' sea',' seal',' shark',' shrew',' skunk',' skyscraper',
          'snail',' snake',' spider',' squirrel',' streetcar',' sunflower',' sweet_pepper',' table',' tank',' telephone',' television',
          ' tiger',' tractor',' train',' trout',' tulip',' turtle',' wardrobe',' whale',' willow_tree',' wolf',' woman',' worm')



"""Fatter net"""

import models
from tqdm import tqdm
 
def train():
 
  best_val_loss = 10000
  best_val_acc = 0

  for epoch in range(0,int(optlist["--epochs"])):  # loop over the dataset multiple times
      
      running_loss = 0.0
      running_accuarcy = 0
      val_loss = 0
      val_acc = 0
      correct = 0
      total = 0
      total_val = 0
      correct_val = 0
      for i, data in enumerate(tqdm(trainloader,desc="Epoch: "+str(epoch)), 0):
          
          
          # get the inputs; data is a list of [inputs, labels]
          net.train()
          inputs, labels = data
          inputs,labels = inputs.to(device), labels.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels).to(device)
          loss.backward()
          optimizer.step()
          
          # print statistics
          with torch.no_grad():
            running_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_accuarcy = 100 * correct // total
          del inputs, labels

      for val_data in validationloader:
        with torch.no_grad():
          val_images, val_labels = val_data
          val_images,val_labels = val_images.to(device), val_labels.to(device)
          #calculate validation
          # val_images, val_labels = val_images.to(device), val_labels.to(device)
          net.eval()
          y_pred = net(val_images)
          val_loss = criterion(y_pred, val_labels).to(device)
          _, val_predicted_label = torch.max(y_pred.data, 1)
          correct_val += (val_predicted_label == val_labels).sum().item()
          total_val += val_labels.size(0)
      val_acc = 100 * correct_val // total_val
      writer.add_scalar("Loss/Train", running_loss, epoch)
      writer.add_scalar("Accuracy/Train", running_accuarcy, epoch)
      writer.add_scalar("Loss/Validation", val_loss, epoch)
      writer.add_scalar("Accuracy/Validation", val_acc, epoch)
      writer.flush()
      sch.step()
      print(f'Epoch{epoch}:      loss: {running_loss:.3f} accuaracy: {running_accuarcy}% val_loss: {val_loss:.3f} val_acc: {val_acc}%\n')
      if val_loss<best_val_loss:
        best_val_loss = val_loss
      if val_acc>best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), "bestfnet.pth")
      elif val_acc==best_val_acc and val_loss<best_val_loss:
        torch.save(net.state_dict(), "bestfnet.pth")
      torch.save(net.state_dict(), "fnet.pth")
      del val_images, val_labels 
        
  print('Finished Training')
  return running_accuarcy, best_val_acc

"""Test"""

def evaluate():
  correct = 0
  total = 0
  y_true=[]
  y_pred=[]
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in tqdm(testloader, desc="evaluation: "):
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          # calculate outputs by running images through the network
          outputs = net(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          y_true.append(labels.cpu().detach().numpy()[:])
          y_pred.append(predicted.cpu().detach().numpy()[:])
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = 100 * correct // total
  print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

  y_true = np.array(y_true, dtype=np.uint8).flatten()
  y_pred = np.array(y_pred, dtype=np.uint8).flatten()
  cf_matrix = confusion_matrix(y_true, y_pred)

  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                      columns = [i for i in classes])
  plt.figure(figsize = (12,7))
  figure = sn.heatmap(df_cm, annot=False).get_figure() 
  figure.savefig('confmat1.png', dpi=400)
  return accuracy




net= models.get_model(optlist["--model"])
#net = nn.DataParallel(net, device_ids=[0,1])
#net.load_state_dict(torch.load("bestfnet.pth"))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
sch = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50,gamma=0.2)
criterion.to(device)

train_acc, val_acc = train()
test_acc = evaluate()
