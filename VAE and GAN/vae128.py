import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    '''
    Variational Autoencoder: Encoder module.
    '''
    def __init__(self,latent_dim):
        '''
        in_channels (int): number of channels of the input image (e.g.: 1 for
                           grayscale, 3 for color images).
        in_dim (int): number of pixels on each row / column of the input image
                      (assumes the images are square).
        latent_dim (int): dimension of the output (latent) space.
        flat_dim (int): flattened dimension after the last conv. layer.
        filters (list of length n_conv): number of filters for each conv.
                                         layer.
        kernel_sizes (list of length n_conv): kernel size for each conv. layer.
        strides (list of length n_conv): strides for each conv. layer.
        paddings (list of length n_conv): zero padding added to the input for
                                          each conv. layer.
        activation (subclass of nn.Module): activation used in all layers,
                                            except in the output (default:
                                            LeakyReLU).
        batch_norm (boolean): if True, batch normalization is applied in every
                              layer before the activation (default: True).
        '''
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        nc=3
        ndf=64
        self.enc_1conv_l1=nn.Sequential(
            nn.Conv2d(nc, ndf, 1, 1, 0, bias=False)
        )

        self.enc_l1=nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc_1conv_l2=nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 1, 1, 0, bias=False)
        )

        self.enc_l2=nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_1conv_l3=nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 1, 1, 0, bias=False)
        )

        self.enc_l3=nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_1conv_l4=nn.Sequential(
            nn.Conv2d(ndf*4, ndf*4, 1, 1, 0, bias=False)
        )

        self.enc_l4=nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_1conv_l5=nn.Sequential(
            nn.Conv2d(ndf*4, ndf*2, 1, 1, 0, bias=False)
        )

        self.enc_l5=nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_l6=nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, latent_dim, 4, 2, 0, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mean_block = nn.Linear(latent_dim, latent_dim)
        self.logvar_block = nn.Linear(latent_dim, latent_dim)
        """
        self.param_init()

    def param_init(self):
        '''Parameters initialization.'''
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)
        """
    def forward(self, input):
            input_res = nn.functional.interpolate(input,scale_factor=1/2)
            out = self.enc_l1(input)+ self.enc_1conv_l1(input_res)

            input_res = nn.functional.interpolate(out,scale_factor=1/2)
            out = self.enc_l2(out)+ self.enc_1conv_l2(input_res)

            input_res = nn.functional.interpolate(out,scale_factor=1/2)
            out = self.enc_l3(out)+ self.enc_1conv_l3(input_res)

            input_res = nn.functional.interpolate(out,scale_factor=1/2)
            out = self.enc_l4(out)+ self.enc_1conv_l4(input_res)

            input_res = nn.functional.interpolate(out,scale_factor=1/2)
            out = self.enc_l5(out)+ self.enc_1conv_l5(input_res)

            out=self.enc_l6(out)
            #print(out.shape)
            out = out.reshape(-1, self.latent_dim)
            #print(out.shape)
            z_mean = self.mean_block(out)
            z_logvar = self.logvar_block(out)
            #print(z_mean.shape)
            """
            z_mean = self.mean_block(h)
            z_logvar = self.logvar_block(h)
            #print("z mean",z_mean.shape)
            #print("z log var",z_logvar.shape)
            return z_mean, z_logvar
            """
            return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim=latent_dim
        nz=latent_dim
        ngf=64

        self.dec_1conv_l1=nn.Sequential(
            nn.Conv2d(nz, ngf*8, 1, 1, 0, bias=False)
        )

        self.dec_l1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)

        )

        self.dec_1conv_l2=nn.Sequential(
            nn.Conv2d(ngf*8, ngf*4, 1, 1, 0, bias=False)
        )

        self.dec_l2 = nn.Sequential(

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #nn.Dropout2d(p=0.5)
        )

        self.dec_1conv_l3=nn.Sequential(
            nn.Conv2d(ngf*4, ngf*2, 1, 1, 0, bias=False)
        )

        self.dec_l3 = nn.Sequential(
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #nn.Dropout2d(p=0.5)
        )

        self.dec_1conv_l4=nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 1, 1, 0, bias=False)
        )

        self.dec_l4 = nn.Sequential(
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.dec_1conv_l5=nn.Sequential(
            nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False)
        )

        self.dec_l5 = nn.Sequential(
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.dec_l6 = nn.Sequential(
            nn.ConvTranspose2d(ngf,     3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.upsample=nn.Upsample(scale_factor=2)
        """
        self.param_init()

    def param_init(self):
        '''Parameters initialization.'''
        for layer in self.modules():
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)
        """
    def forward(self, input):
            #print("decoder input",input.shape)

            #input = self.upsample(input.view(256,100,1,-1))

            x = self.dec_l1(input.view(-1,self.latent_dim,1,1))

            input = self.upsample(x)
            x = self.dec_l2(x) + self.dec_1conv_l2(input)

            input = self.upsample(x)
            x = self.dec_l3(x) + self.dec_1conv_l3(input)

            input = self.upsample(x)
            x = self.dec_l4(x) + self.dec_1conv_l4(input)

            input = self.upsample(x)
            x = self.dec_l5(x) + self.dec_1conv_l5(input)

            output = self.dec_l6(x)

            return output


class VAE(nn.Module):
    '''
    Variational Autoencoder: encoder + sampling + decoder.
    '''
    def __init__(self,latent_dim):
        '''
        in_dim (int): number of pixels on each row / column of the images
                      (assumes the images are square).
        in_channels (int): number of channels of the images (e.g.: 1 for
                           grayscale, 3 for color images).
        latent_dim (int): dimension of the latent space.
        filters (list of length n_conv): number of filters for each conv.
                                         layer.
        kernel_sizes (list of length n_conv): kernel size for each conv.
        strides (list of length n_conv): strides for each conv. layer.
        activation (nn.Module): activation used in all layers (default:
                                LeakyReLU).
        out_activation (subclass of nn.Module): activation used in the output
                                                layer (default: Tanh).
        batch_norm (boolean): if True, batch normalization is applied in every
                              layer before the activation (default: True).
        '''
        super(VAE, self).__init__()



        #self.latent_dim = latent_dim



        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)


    def sample(self, z_mean, z_logvar):
        '''Parameters initialization.'''
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(.5*z_logvar) * eps

        return z

    def forward(self, X):
        '''Forward pass.'''
        z_mean, z_logvar = self.encoder(X)

        z = self.sample(z_mean, z_logvar)
        #print(z.shape)
        Xrec = self.decoder(z)
        #Xrec=self.sigm(Xrec)
        return Xrec, z_mean, z_logvar



def vae_loss(Xrec, X, z_mean, z_logvar, kl_weight=1e-3):
    '''
    Loss function of the Variational Autoencoder.
    Includes a hyperparameter kl_weight to control the relative weight of
    each term in the loss.
    '''
    reconst_ls = F.mse_loss(Xrec, X)
    #reconst_ls = F.binary_cross_entropy(Xrec.view(-1,64*64*3), X.view(-1,64*64*3), size_average=False)
    kl_ls = torch.mean(-.5*torch.sum(1 + z_logvar - z_mean**2
                                     - torch.exp(z_logvar), dim=1), dim=0)

    loss = reconst_ls #+ kl_weight * kl_ls

    return loss, reconst_ls, kl_ls
