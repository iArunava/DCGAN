import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.utils as utils

cudnn.benchmark = True
print ('[INFO]Set cudnn to True.')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 128
z_size = 100

transform = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
                              
train_data = datasets.CIFAR10('./', download=True, transform=transform)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

def init_weight(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.2)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
    
# Discriminator

# Probably a VGG16 or VGG19 for Simple Image Classification pretrained on ImageNet

class Discriminator(nn.Module):
    
    def __init__(self, inhw, c1_channels=64, c2_channels=128, c3_channels=256,
                 c4_channels=512, i_channels_in_2=True):
        '''
        The constructor method for the Discriminator class
        
        Arguments:
        - inhw : The number of 
        - c1_channels : the number of output channels from the
                        first Convolutional Layer [Default - 128]
                        
        - c2_channels : the number of output channels from the
                        second Convolutional Layer [Default - 256]
                        
        - c3_channels : the number of output channels from the
                        third Convolutional Layer [Default - 512]
        
        - i_channels_in_2 : Increase the number of channels by 2
                        in each layer.
        '''
        
        super().__init__()
        
        # Define the class variables
        self.c1_channels = c1_channels
        
        if i_channels_in_2:
            self.c2_channels = self.c1_channels * 2
            self.c3_channels = self.c2_channels * 2
            self.c4_channels = self.c3_channels * 2
        else:
            self.c2_channels = c2_channels
            self.c3_channels = c3_channels
            self.c4_channels = c4_channels
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.c1_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=self.c1_channels,
                               out_channels=self.c2_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        
        self.bnorm2 = nn.BatchNorm2d(num_features=self.c2_channels)
        
        self.conv3 = nn.Conv2d(in_channels=self.c2_channels,
                               out_channels=self.c3_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        
        self.bnorm3 = nn.BatchNorm2d(num_features=self.c3_channels)
        
        self.conv4 = nn.Conv2d(in_channels=self.c3_channels,
                               out_channels=self.c4_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)
        
        self.bnorm4 = nn.BatchNorm2d(num_features=self.c4_channels)
        
        self.conv5 = nn.Conv2d(in_channels=self.c4_channels,
                               out_channels=1,
                               kernel_size=4,
                               padding=0,
                               stride=1,
                               bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, img):
        '''
        The method for the forward pass in the network
        
        Arguments;
        - img : a torch.tensor that is of the shape N x C x H x W
                where, N - the batch_size
                       C - the number of channels
                       H - the height
                       W - the width
       
       Returns:
       - out : the output of the Discriminator 
               whether the passed image is real /fake
        '''
        
        #print (img.shape)
        
        batch_size = img.shape[0]
        
        x = self.lrelu(self.conv1(img))
        x = self.lrelu(self.bnorm2(self.conv2(x)))
        x = self.lrelu(self.bnorm3(self.conv3(x)))
        x = self.lrelu(self.bnorm4(self.conv4(x)))
        x = self.conv5(x)
        
        x = self.sigmoid(x)
        
        return x.view(-1, 1).squeeze()
      
    def out_shape(self, inp_dim, kernel_size=4, padding=1, stride=2):
        return ((inp_dim - kernel_size + (2 * padding)) // stride) + 1
        
class Generator(nn.Module):
    def __init__(self, ct1_channels=512, ct2_channels=256,
                 ct3_channels=128, ct4_channels=64, d_channels_in_2=False):
        
        '''
        The contructor class for the Generator
        
        Arguments:
        - zin_channels: ###
        
        - ct1_channels: The number of output channels for the
                        first ConvTranspose Layer. [Default - 1024]
        
        - ct2_channels: The number of putput channels for the
                        second ConvTranspose Layer. [Default - 512]
                        
        - ct3_channels: The number of putput channels for the
                        third ConvTranspose Layer. [Default - 256]
                        
        - ct4_channels: The number of putput channels for the
                        fourth ConvTranspose Layer. [Default - 128]
                        
        - d_channnels_in_2 : Decrease the number of channels 
                        by 2 times in each layer.
                        
        '''
        super().__init__()
        
        # Define the class variables
        self.ct1_channels = ct1_channels
        self.pheight = 4
        self.pwidth = 4
        
        if d_channels_in_2:
            self.ct2_channels = self.ct1_channels // 2
            self.ct3_channels = self.ct2_channels // 2
            self.ct4_channels = self.ct3_channels // 2
        else:
            self.ct2_channels = ct2_channels
            self.ct3_channels = ct3_channels
            self.ct4_channels = ct4_channels
        
        self.convt_0 = nn.ConvTranspose2d(in_channels=z_size,
                                          out_channels=self.ct1_channels,
                                          kernel_size=4,
                                          padding=0,
                                          stride=1,
                                          bias=False)
        
        self.bnorm0 = nn.BatchNorm2d(self.ct1_channels)
        
        self.convt_1 = nn.ConvTranspose2d(in_channels=self.ct1_channels,
                                          out_channels=self.ct2_channels,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        
        self.bnorm1 = nn.BatchNorm2d(num_features=self.ct2_channels)
        
        self.convt_2 = nn.ConvTranspose2d(in_channels=self.ct2_channels,
                                          out_channels=self.ct3_channels,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        
        self.bnorm2 = nn.BatchNorm2d(num_features=self.ct3_channels)
        
        self.convt_3 = nn.ConvTranspose2d(in_channels=self.ct3_channels,
                                          out_channels=self.ct4_channels,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        
        self.bnorm3 = nn.BatchNorm2d(num_features=self.ct4_channels)
        
        self.convt_4 = nn.ConvTranspose2d(in_channels=self.ct4_channels,
                                          out_channels=3,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        '''
        The method for the forward pass for the Generator
        
        Arguments:
        - z : the input random uniform vector sampled from uniform distribution
        
        Returns:
        - out : The output of the forward pass through the network
        '''
        
        # Project the input z and reshape
        x = self.relu(self.bnorm0(self.convt_0(z)))
        #print (x.shape)
        x = self.relu(self.bnorm1(self.convt_1(x)))
        x = self.relu(self.bnorm2(self.convt_2(x)))
        x = self.relu(self.bnorm3(self.convt_3(x)))
        out = self.tanh(self.convt_4(x))
        
        return out
        
D = Discriminator(64).to(device)
#D = Discriminator(1)
D.apply(init_weight)
G = Generator().to(device)
#G = Generator(1)
G.apply(init_weight)

print (D)
print ()
print (G)

criterion = nn.BCELoss()

# Optimizers

d_lr = 0.0002
g_lr = 0.0002

d_opt = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_opt = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train loop
#84

p_every = 1
t_every = 1
e_every = 1
s_every = 10
epochs = 25

real_label = 1
fake_label = 0

train_losses = []
eval_losses = []

#dcgan.train()
#D.train()
#G.train()

for e in range(epochs):
    
    td_loss = 0
    tg_loss = 0
    
    for batch_i, (real_images, _) in enumerate(trainloader):
        
        # Scaling image to be between -1 and 1
        #real_images = scale(real_images)
        
        real_images = real_images.to(device)
        
        batch_size = real_images.size(0)
        
        # Update te Discrmininator every 3 epochs

        #### Train the Discriminator ####

        #d_opt.zero_grad()
        D.zero_grad()

        #print (real_images.shape)
        d_real = D(real_images)
        
        label = torch.full((batch_size,), real_label, device=device)
        r_loss = criterion(d_real, label)
        #r_loss = real_loss(d_real)
        #r_loss.backward()


        z = torch.randn(batch_size, z_size, 1, 1, device=device)
        #z = np.random.uniform(-1, 1, size=(batch_size, z_size, 1, 1))
        #z = torch.from_numpy(z).float().cuda()

        fake_images = G(z)
        
        label.fill_(fake_label)
        
        d_fake = D(fake_images.detach())
        
        f_loss = criterion(d_fake, label)
        #f_loss = fake_loss(d_fake)
        f_loss.backward()

        d_loss = r_loss + f_loss
        #d_loss.backward()

        #td_loss += d_loss.item()

        d_opt.step()


        #### Train the Generator ####
        G.zero_grad()
        #g_opt.zero_grad()
        
        label.fill_(real_label)
        #z = torch.randn(batch_size, z_size, 1, 1, device=device)
        #z = np.random.uniform(-1, 1, size=(batch_size, z_size, 1, 1))
        #z = torch.from_numpy(z).float().cuda()
        #fake_images = G(z)
        d_fake2 = D(fake_images)
        
        #label = torch.full((batch_size,), real_label, device=device)
        
        g_loss = criterion(d_fake2, label)
        #g_loss = real_loss(d_fake)
        g_loss.backward()
        
        #tg_loss += g_loss.item()
        
        g_opt.step()
        
        if batch_i % p_every == 0:
            print ('Epoch [{:5d} / {:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'. \
                    format(e+1, epochs, d_loss, g_loss))
            
    train_losses.append([td_loss, tg_loss])
    
    if e % s_every == 0:
        d_ckpt = {
            'model_state_dict' : D.state_dict(),
            'opt_state_dict' : d_opt.state_dict()
        }

        g_ckpt = {
            'model_state_dict' : G.state_dict(),
            'opt_state_dict' : g_opt.state_dict()
        }

        torch.save(d_ckpt, 'd-nm-{}.pth'.format(e))
        torch.save(g_ckpt, 'g-nm-{}.pth'.format(e))
    
    utils.save_image(fake_images.detach(), 'fake_{}.png'.format(e), normalize=True)
        
print ('[INFO] Training Completed successfully!')

fig, ax = plt.subplots()
losses = np.array(train_losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title('Training Losses')
plt.legend()
plt.show()
