import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

################

channles = 1
img_size = 28
img_shape = (channles, img_size, img_size)

latent_dim = 100

cuda = True if torch.cuda.is_available() else False

################

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
    

################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    

################

adversarial_loss = torch.nn.BCELoss()

################

generator = Generator()
discriminator = Discriminator()

print(generator)
print(discriminator)


if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

################

import pandas as pd
from torch.utils.data import Dataset

class DatasetMNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28, 1)
        # img = np.concatenate((img, img, img), axis=2)
        label = self.data.iloc[idx, 0]
        # sample = {'image': img, 'label': label}

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

################

train = pd.read_csv('data/train.csv')

# print(train)

################

for index in range(1,6):
    temp_image = train.iloc[index, 1:].values.astype(np.uint8).reshape(28, 28, 1)
    temp_label = train.iloc[index, 0]
    print('Shape of image: ', temp_image.shape)
    print('Label of image: ', temp_label)


dataset = DatasetMNIST('data/train.csv', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])]))

temp_image, _ = dataset.__getitem__(0)
print(temp_image.size())

print(temp_image.max(), temp_image.min())


batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

temp_images, _ = next(iter(dataloader))
print('Images shape on batch size = {}'.format(temp_images.size()))

################

b1 = 0.5
b2 = 0.999

lr = 0.0002

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

################

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

from tqdm import tqdm

import matplotlib.pyplot as plt

n_epochs = 50


for epoch in range(n_epochs):
    for i,(imgs,_) in enumerate(tqdm(dataloader)):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = imgs.type(Tensor)

        optimizer_G.zero_grad()

        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        sample_z_in_train = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))
        sample_gen_imgs_in_train = generator(sample_z_in_train).detach().cpu()

        if ((i+1)%500) == 0:
            nrows = 1
            ncols = 5
            fig, axes = plt.subplots(nrows, ncols, figsize=(8, 2))
            plt.suptitle('EPOCH : {} | BATCH(ITERATION) : {}'.format(epoch+1, i+1))
            for ncol in range(ncols):
                axes[ncol].imshow(sample_gen_imgs_in_train.permute(0,2,3,1)[ncol], cmap='gray')
                axes[ncol].axis('off')
            plt.show()

    print('Epoch : {} | Generator loss : {} | Discriminator loss : {}'.format(epoch+1, g_loss, d_loss))



