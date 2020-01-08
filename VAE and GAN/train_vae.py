"""
This code is taken from:https://github.com/dpernes/vae

"""
import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
import numpy as np
import sys
import argparse
from vae128 import VAE, vae_loss
from imgdataset import ImgDataset
from utils import imsave
import os
from torchvision.utils import save_image

from torch.optim import lr_scheduler


import torch.utils.data

from torch.autograd import Variable



parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--data-path', type=str,
                    default='../data/Selfie',
                    help='path for the images dir')
parser.add_argument('--img-crop', type=int, default=148,
                    help='size for center cropping (default: 148)')
parser.add_argument('--img-resize', type=int, default=128,
                    help='size for resizing (default: 64)')
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--valid-split', type=float, default=.2,
                    help='fraction of data for validation (default: 0.2)')
parser.add_argument('--kl-weight', type=float, default=1e-3,
                    help='weight of the KL loss (default: 1e-3)')
parser.add_argument('--filters', type=str, default='64, 128, 256, 512',
                    help=('number of filters for each conv. layer (default: '
                          + '\'64, 128, 256, 512\')'))
parser.add_argument('--kernel-sizes', type=str, default='3, 3, 3, 3',
                    help=('kernel sizes for each conv. layer (default: '
                          + '\'3, 3, 3, 3\')'))
parser.add_argument('--strides', type=str, default='2, 2, 2, 2',
                    help=('strides for each conv. layer (default: \'2, 2, 2, '
                          + '2\')'))
parser.add_argument('--latent-dim', type=int, default=1024,
                    help='latent space dimension (default: 128)')
parser.add_argument('--batch-norm', type=int, default=1,
                    help=('whether to use or not batch normalization (default:'
                          + ' 1)'))
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
args = parser.parse_args()
args.filters = [int(item) for item in args.filters.split(',')]
args.kernel_sizes = [int(item) for item in args.kernel_sizes.split(',')]
args.strides = [int(item) for item in args.strides.split(',')]
args.batch_norm = bool(args.batch_norm)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
np.random.seed(args.seed)

is_cuda=True

def train(vae,optimizer, train_loader, n_epochs,image_save_file,kl_weight=1e-4,
          test_loader=None, n_gen=0,schedular=None):


    for epoch in range(n_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):

            data = Variable(data)
            if is_cuda:
                data = data.cuda()
            optimizer.zero_grad()

            #if data.size(0)==256:

            Xrec,z_mean,z_logvar = vae(data)

            # loss, backward pass and optimization step

            loss, reconst_loss, kl_loss = vae_loss(Xrec, data, z_mean, z_logvar,
                                                   kl_weight=kl_weight)

            loss.backward()

            train_loss += loss.data.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        schedular.step()

        vae.eval()

        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if is_cuda:# and data.size(0)==256:
                    data = data.cuda()
                    data = Variable(data)
                    Xrec, z_mean, z_logvar = vae(data)
                    test_loss += vae_loss(Xrec, data, z_mean, z_logvar,
                                           kl_weight=kl_weight)[0]

                if i == 0:
                    #imsave(data, './dog_cat/valid_orig{}.png'.format(epoch))
                    #imsave(Xrec, './dog_cat/valid_rec{}.png'.format(epoch))
                    n = min(data.size(0), 16)
                    comparison = torch.cat([((data+ 1.)/2.)[:n],
                                          ((Xrec.view(args.batch_size, 3, 128, 128)+1.)/2.)[:n]])
                    save_image(comparison.data.cpu(), '%s/%03d.png' % (image_save_file, epoch), nrow=n)

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))
            torch.save(vae.state_dict(), './models/vae_zoom'+str(epoch)+'.pth')



SetRange = transforms.Lambda(lambda X: 2*X - 1.)
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(0,scale=(1.3,2)),
                                #transforms.CenterCrop(200),
                                transforms.Resize((128,
                                                   128)),
                                transforms.ToTensor(),
                                SetRange])



data_dir="../../img_celeba"
#data_dir="../../Processed_data/Olulu_Casia_vae"

train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, "train"),
               transform=transform), batch_size=args.batch_size,
                                           shuffle=True, num_workers=4
                                          )

valid_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, "val"),
               transform=transform), batch_size=args.batch_size,
                                           shuffle=False, num_workers=4
                                           )




vae = VAE(

          latent_dim=1024,
        ).to(DEVICE)


model_load="models/vae_kl8.pth"

checkpoint = torch.load(model_load)
vae.load_state_dict(checkpoint)
#vae=torch.load(model_load)
print(vae)



optimizer = optim.Adam(vae.parameters(),
                       lr=1e-4,
                       weight_decay=0.)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

train(vae,optimizer, train_loader, args.epochs,"vae_zoom",
      test_loader=valid_loader, n_gen=args.batch_size,schedular=exp_lr_scheduler)
