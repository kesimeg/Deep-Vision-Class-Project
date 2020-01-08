"""
This code trains on dataset generated with dataset creator code.
This code does 9-fold cross validation on the dataset and writes the accuracy and loss values in to cvs file.
Resnet18 model is used
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datetime  import datetime

data_dir = '../Processed_data/Olulu_Casia_one_subject_last_three'


torch.manual_seed(0)
np.random.seed(0)

class random_noise(object):
    def __call__(self, image):
        noise=torch.rand_like(image)/10.
        return image+noise



class illumination_change(object):
    def __call__(self, image):
        mean_val=torch.mean(image)
        if mean_val > 0.59:
            r1,r2=1.,2.
        else:
            r1,r2=0.7,1.

        div=(r1 - r2) * torch.rand(1) + r2
        return image/div

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((224,
                                                   224)),
                                transforms.RandomAffine(degrees=0,translate=(0.05,0.1)),
                                transforms.ToTensor(),
                                random_noise(),
                                illumination_change(),
                                ])



dataset_train0=datasets.ImageFolder(os.path.join(data_dir, "Set0"),transform=transform)
dataset_train1=datasets.ImageFolder(os.path.join(data_dir, "Set1"),transform=transform)
dataset_train2=datasets.ImageFolder(os.path.join(data_dir, "Set2"),transform=transform)
dataset_train3=datasets.ImageFolder(os.path.join(data_dir, "Set3"),transform=transform)
dataset_train4=datasets.ImageFolder(os.path.join(data_dir, "Set4"),transform=transform)
dataset_train5=datasets.ImageFolder(os.path.join(data_dir, "Set5"),transform=transform)
dataset_train6=datasets.ImageFolder(os.path.join(data_dir, "Set6"),transform=transform)
dataset_train7=datasets.ImageFolder(os.path.join(data_dir, "Set7"),transform=transform)
dataset_train8=datasets.ImageFolder(os.path.join(data_dir, "Set8"),transform=transform)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)






def imshow(inp, title=None):
    #"Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    #plt.pause(1)  # pause a bit so that plots are updated


"""
# Get a batch of training data

inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[0:6]) #belli bir bölümünü ekrana bas (resmin son halini görmek için)

imshow(out)#, title=[class_names[x] for x in classes][0:6])
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))

        #list(res.children())[0]


        resnet_firstlayer=list(models.resnet18(pretrained = True).children())[0] #load just the first conv layer
        resnet=torch.nn.Sequential(*(list(models.resnet18(pretrained = True).children())[1:-2])) #load upto the classification layers except first conv layer and pool
       
        """
        #This part can be used to turn the inputs in to grayscale, first layers channels are addded to each other
        w1=resnet_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=resnet_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=resnet_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional


        first_conv=nn.Conv2d(1, 64, 7,stride=(2,2),padding = (3,3)) #create a new conv layer
        first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        #first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias


        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv  layer
        """
        
        self.first_convlayer=resnet_firstlayer
        self.resnet =nn.Sequential(resnet)

        self.fc1 = nn.Linear(7*7*512, 1000)
        self.b_norm=nn.BatchNorm1d(1000)
        self.dropout=nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(1000, 6)

        self.classify=nn.Sequential(
            self.dropout,
            self.fc1,
            nn.ReLU(),
            self.b_norm,
            self.fc2
        )
    def forward(self, x):
        x=self.first_convlayer(x)
        x = self.resnet(x)
        x = x.view(-1, 7 * 7 * 512)
        x=self.classify(x)
        return x



def train_model(model, criterion, optimizer, num_epochs=25,checkp_epoch=0):
    since = time.time()


    my_file=open(plot_file, "a")


    pbar=tqdm(range(checkp_epoch,num_epochs))
    for epoch in pbar:

        model.train()

        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()


            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


                loss.backward()
                optimizer.step()


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / train_set_size
        train_acc = running_corrects.double() / train_set_size

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / valid_set_size
        val_acc = running_corrects.double() / valid_set_size

        torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss
             },checkpoint_file)


        data = {'epoch': epoch,
        'train_loss': train_loss,
        'train_acc':train_acc.item(),
        'val_loss':val_loss,
        'val_acc':val_acc.item()
        }
        df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
        df.to_csv(my_file, header=False,index=False)

        pbar.set_description("train acc {:.3} loss {:.4} val acc {:.3} loss {:.4}".format(train_acc,train_loss,val_acc,val_loss))
    time_elapsed = time.time() - since

    return model,val_acc,train_acc




# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


data_set_list=[dataset_train0,dataset_train1, dataset_train2,dataset_train3,dataset_train4,
dataset_train5,dataset_train6,dataset_train7,dataset_train8]


num_epochs=100

test_mean=0.
train_mean=0.

for i in range(0,len(data_set_list)):
    now=datetime.now()
    checkpoint_file="emotion"+now.strftime("%d_%m_%Y_%H:%M:%S")+".pt"
    plot_file="emotion_plot"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"
    print(plot_file)
    dataset_train=torch.utils.data.ConcatDataset(data_set_list[:i]+data_set_list[i+1:])
    #https://pytorch.org/docs/master/data.html#torch.utils.data.ConcatDataset
    train_loader = torch.utils.data.DataLoader(dataset_train,
                    batch_size=128,
                                               shuffle=True, num_workers=4)

    valid_loader = torch.utils.data.DataLoader(data_set_list[i],
                    batch_size=128,
                                               shuffle=False, num_workers=4)

    model_ft = Net()
    model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(),lr=0.00001)

    train_set_size = len(dataset_train)
    valid_set_size = len(data_set_list[i])

    print("Train set size:",train_set_size)
    print("Test set size:",valid_set_size)
    model_ft,val_acc,train_acc = train_model(model_ft, criterion, optimizer_ft,
                   num_epochs=num_epochs)
    train_mean+=train_acc
    test_mean+=val_acc
print("Test_mean ",test_mean/len(data_set_list))
print("Train_mean ",train_mean/len(data_set_list))
