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
        """
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
        """
        self.classify=nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.b_norm,
            self.dropout,
            self.fc2
        )
        """
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
        #print(x.shape)
        x = x.view(-1, 7 * 7 * 512)
        """
        x = F.relu(self.fc1(x))
        x = self.b_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        """
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



#model_path="./Facenet_drop0.9_lr=0.00002_batch=128/emotion06_01_2020_20:37:58.pt"

#model_path="./Facenet_drop0.9_lr=0.00002_batch=128/emotion06_01_2020_20:45:42.pt"

model_path="./Resnet_rgb_lr=0.00001_drop=0.8_batch_128/emotion06_01_2020_17:29:28.pt"
checkpoint_file=torch.load(model_path)["model_state_dict"]

#model=models.resnet50(pretrained=True)
"""
model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=6,
    dropout_prob=0.9
)
"""

model=Net()
model.load_state_dict(checkpoint_file)


model.eval()



"""
train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("load", transforms.Compose([
            #transforms.RandomSizedCrop(224),
            transforms.Resize([160,160]),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            random_noise(),
            illumination_change(),
        ])),
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)


tensor_img=torch.tensor(image)
tensor_img=tensor_img.permute((2,1,0))
print(tensor_img.size())
tensor_img=normalize(tensor_img)
tensor_img=tensor_img.view([1,3,224,224])




dataiter = iter(train_loader)
image, labels = dataiter.next()
print(image.size())

"""
#images=normalize(images[0,:,:,:])

#images=torch.unsqueeze(images,0)
#images+=torch.rand(3,224,224)

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
#print(predict)

#print(torch.sum(output))
#print(torch.sum(predict))




def color_up(arr):
    a=np.where(arr>10**-2,arr+0.2,arr)
    #a=np.where(arr<0.8,0,arr)
    return a



grad_img=np.array(grad_img)[0,:,:]

difference_img=np.array(image_orig).astype("float32")



"""
xmax, xmin = grad_img.max(), grad_img.min()
grad_img= (grad_img - xmin)/(xmax - xmin)
"""
print(grad_img)

print("Maximum of grad",np.max(grad_img))
grad_img=color_up(grad_img)

difference_img[:,:,0]*=grad_img
difference_img[:,:,1]*=grad_img
difference_img[:,:,2]*=grad_img
print("Prediction",predict)
#print(type(grad_img),grad_img.shape)
print("Maxindex and maximum value of prediction",index,maxim)

plt.imshow(grad_img,cmap="gray")#,vmin=0, vmax=1)
plt.show()
plt.imshow(difference_img)
plt.show()
