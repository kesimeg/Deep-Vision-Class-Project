"""
This  code uses only landmark data. 
Some transform functions are based on the transforms given in Pytorch tutorials:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms,utils
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datetime  import datetime
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


data_dir = '../Processed_data/Olulu_Casia_one_subject_last_three'


def show_landmarks(image, landmarks,label):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks , label= sample['image'], sample['landmarks'],sample["label"]

        h, w = image.shape[:2]
        """
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
        """
        new_h, new_w = self.output_size,self.output_size
        #print("new h,w before int",new_h,new_w)
        #new_h, new_w = int(new_h), int(new_w)
        #print("new h,w",new_h,new_w)
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks,"label": label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks , label= sample['image'], sample['landmarks'],sample["label"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #print(image.shape)
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                 'label':label}



class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
        img_name = self.landmarks_frame.iloc[idx, 0]
        image = io.imread(img_name)
        label = self.landmarks_frame.iloc[idx, 1]
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, "label":label}

        if self.transform:
            sample = self.transform(sample)

        return sample




# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch[0:8])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(8):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

class random_noise(object):
    def __call__(self, sample):
        image, landmarks , label= sample['image'], sample['landmarks'],sample["label"]
        noise=torch.rand_like(image)/10.
        return {'image': image+noise,
                'landmarks': landmarks,
                 'label':label}



class illumination_change(object):
    def __call__(self, sample):
        image, landmarks , label= sample['image'], sample['landmarks'],sample["label"]
        mean_val=torch.mean(image)
        if mean_val > 0.59:
            r1,r2=1.,2.
        else:
            r1,r2=0.7,1.

        div=(r1 - r2) * torch.rand(1) + r2

        return {'image': image/div,
                'landmarks': landmarks,
                 'label':label}

class horizontal_flip(object):
    def __init__(self, image_dim):
        self.image_dim = image_dim
    def __call__(self, sample):
        image, landmarks , label= sample['image'], sample['landmarks'],sample["label"]

        if torch.rand(1) > 0.5:
            return {'image': image,
                    'landmarks': landmarks,
                     'label':label}
        else:

            image =  np.flip(image,axis=1).copy()#np.flip(image,0)#image.transpose(Image.FLIP_LEFT_RIGHT)

            landmarks[:,0]=self.image_dim-landmarks[:,0]
            return {'image': image,
                'landmarks': landmarks,
                 'label':label}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.landmark_linear= nn.Sequential(
        nn.Linear(68*2, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.7),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 6))
        
    def forward(self,landmark):


        out = self.landmark_linear(landmark.view(-1,68*2))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])

dataset_train0=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set0.csv',root_dir='data/faces/',transform=transform_var)
dataset_train1=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set1.csv',root_dir='data/faces/',transform=transform_var)
dataset_train2=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set2.csv',root_dir='data/faces/',transform=transform_var)
dataset_train3=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set3.csv',root_dir='data/faces/',transform=transform_var)
dataset_train4=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set4.csv',root_dir='data/faces/',transform=transform_var)
dataset_train5=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set5.csv',root_dir='data/faces/',transform=transform_var)
dataset_train6=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set6.csv',root_dir='data/faces/',transform=transform_var)
dataset_train7=dataset_train = FaceLandmarksDataset(csv_file='Landmark_data/Set7.csv',root_dir='data/faces/',transform=transform_var)

dataset_train=torch.utils.data.ConcatDataset([dataset_train0,dataset_train1, dataset_train2,dataset_train3,dataset_train4,
dataset_train5,dataset_train6,dataset_train7])
#https://pytorch.org/docs/master/data.html#torch.utils.data.ConcatDataset



dataset_valid = FaceLandmarksDataset(csv_file='Landmark_data/Set8.csv',
                                           root_dir='data/faces/',
                                           transform=transform_var)
train_loader = DataLoader(dataset_train, batch_size=128,
                        shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset_valid, batch_size=128,
                        shuffle=True, num_workers=4)


train_set_size = len(dataset_train)
valid_set_size = len(dataset_valid)

print("Train set size:",train_set_size)
print("Test set size:",valid_set_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)



#class_names = dataset_train.classes


def imshow(inp, title=None):
    #"Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    #plt.pause(1)  # pause a bit so that plots are updated



# Get a batch of training data

sample = next(iter(train_loader))


inputs, landmark, classes =  sample["image"], sample['landmarks'],sample["label"]
# Make a grid from batch
out = torchvision.utils.make_grid(inputs[0:6]) #belli bir bölümünü ekrana bas (resmin son halini görmek için)

imshow(out)#, title=[class_names[x] for x in classes][0:6])

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

def train_model(model, criterion, optimizer, num_epochs=25,checkp_epoch=0):
    since = time.time()

    my_file=open(plot_file, "a")


    pbar=tqdm(range(checkp_epoch,num_epochs))
    for epoch in pbar:

        model.train()  

        running_loss = 0.0
        running_corrects = 0

        for sample in train_loader:
            inputs = sample["image"]
            labels = sample["label"]
            landmarks = sample["landmarks"]


            inputs = inputs.to(device)
            labels = labels.to(device)
            landmarks = landmarks.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(landmarks)
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
            for sample in valid_loader:
                inputs = sample["image"]
                labels = sample["label"]
                landmarks = sample["landmarks"]

                inputs = inputs.to(device)
                labels = labels.to(device)
                landmarks = landmarks.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(landmarks)
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
        #print()
        pbar.set_description("train acc {:.3} loss {:.4} val acc {:.3} loss {:.4}".format(train_acc,train_loss,val_acc,val_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model



model_ft = Net().to(device)
model_ft.apply(weights_init)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(),lr=2*1e-4)


# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)


now=datetime.now()

checkpoint_file="emotion"+now.strftime("%d_%m_%Y_%H:%M:%S")+".pt"
plot_file="emotion_plot"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


num_epochs=400
if os.path.exists(checkpoint_file):
        print("Model loaded")
        checkpoint = torch.load(checkpoint_file)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        model_ft.train()
        model_ft = train_model(model_ft, criterion, optimizer_ft,
                       num_epochs=num_epochs,checkp_epoch=checkpoint['epoch']+1)
else:
    model_ft = train_model(model_ft, criterion, optimizer_ft,
                       num_epochs=num_epochs)
