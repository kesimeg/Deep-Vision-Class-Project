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


import os
import sys
import numpy as np
from skimage import io, transform
from inception_resnet_v1 import InceptionResnetV1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        #newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))

        #list(res.children())[0]


        resnet_firstlayer=list(models.resnet18(pretrained = True).children())[0] #load just the first conv layer
        resnet=torch.nn.Sequential(*(list(models.resnet18(pretrained = True).children())[1:-2])) #load upto the classification layers except first conv layer and pool
        
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

def random_noise(image):
        noise=torch.rand_like(image)/10.
        return image+noise



def illumination_change(image):
        mean_val=torch.mean(image)
        if mean_val > 0.59:
            r1,r2=1.,2.
        else:
            r1,r2=0.7,1.

        div=(r1 - r2) * torch.rand(1) + r2
        return image/div

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]

filename = dir_path +'/' +image_path


image = io.imread(filename)

image = transform.resize(image, (224, 224))
image_orig=image
image = image.transpose((2, 0, 1))
image =  torch.from_numpy(image).float()
image = image.unsqueeze(0)



model_path="./Resnet_rgb_lr=0.00001_drop=0.8_batch_128/emotion06_01_2020_17:29:28.pt"
checkpoint_file=torch.load(model_path)["model_state_dict"]

model=Net()
model.load_state_dict(checkpoint_file)


model.eval()

grad_img=torch.zeros(1,224,224)
for i in range(0,500):
    image_copy = image.clone()
    image_copy=random_noise(image_copy)
    image_copy.requires_grad=True
    criterion = nn.CrossEntropyLoss()


    output=model(image_copy) #normalize edilmemiş
    predict=torch.softmax(output,1)
    class_num=0
    label=torch.tensor([class_num])
    loss = criterion(output,label)

    maxim,index=torch.max(predict,1)
    loss.backward()

    grad2=image_copy.grad
    grad2=torch.abs(grad2)
    grad2,_=torch.max(grad2,dim=1)
    grad_img+=grad2
grad_img=grad_img/500

def color_up(arr):
    a=np.where(arr>10**-2,arr+0.2,arr)
    #a=np.where(arr<0.8,0,arr)
    return a



grad_img=np.array(grad_img)[0,:,:]

difference_img=np.array(image_orig).astype("float32")

print(grad_img)

print("Maximum of grad",np.max(grad_img))
grad_img=color_up(grad_img)

difference_img[:,:,0]*=grad_img
difference_img[:,:,1]*=grad_img
difference_img[:,:,2]*=grad_img
print("Prediction",predict)

print("Maxindex and maximum value of prediction",index,maxim)

plt.imshow(grad_img,cmap="gray")#,vmin=0, vmax=1)
plt.show()
plt.imshow(difference_img)
plt.show()
