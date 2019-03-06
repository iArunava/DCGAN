import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import os

from models.Discriminator import Discriminator
from models.Generator import Generator
from utils import init_weight
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def train(FLAGS):
    # Define the hyperparameters
    p_every = FLAGS.p_every
    s_every = FLAGS.s_every
    epochs = FLAGS.epochs
    dlr = FLAGS.dlr
    glr = FLAGS.glr
    beta1 = FLAGS.beta1
    beta2 = FLAGS.beta2
    z_size = FLAGS.zsize
    batch_size = FLAGS.batch_size
    rh = FLAGS.resize_height
    rw = FLAGS.resize_width
    d_path = FLAGS.dataset_path
    d_type = FLAGS.dataset_type
    
    # Preprocessing Data
    transform = transforms.Compose([transforms.Resize((rh, rw)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    
    if FLAGS.dataset_path == None:
        if d_type == "cars":
            if not os.path.exists('./datasets/cars_train'):
                os.system('sh ./datasets/dload.sh cars')
            d_path = './datasets/cars_train/'

        elif d_type == "flowers":
            if not os.path.exists('./datasets/flowers/'):
                os.system('sh ./datasets/dload.sh flowers')
            d_path = './datasets/flowers/'

        elif d_type == "dogs":
            if not os.path.exists('./datasets/jpg'):
                os.system('sh ./datasets/dload.sh dogs')
            d_path = './datasets/jpg/'
            
    train_data = datasets.ImageFolder(d_path, transform=transform)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define the D and G
    dis = Discriminator(64)
    gen = Generator()

    # Apply weight initialization
    dis.apply(init_weight)
    gen.apply(init_weight)

    # Define the loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    d_opt = optim.Adam(dis.parameters(), lr=dlr, betas=(beta1, beta2))
    g_opt = optim.Adam(gen.parameters(), lr=glr, betas=(beta1, beta2))

    # Train loop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_losses = []
    eval_losses = []

    dis.to(device)
    gen.to(device)
    
    real_label = 1
    fake_label = 0

    for e in range(epochs):

        td_loss = 0
        tg_loss = 0

        for batch_i, (real_images, _) in enumerate(trainloader):

            real_images = real_images.to(device)

            batch_size = real_images.size(0)

            #### Train the Discriminator ####

            d_opt.zero_grad()
            
            d_real = dis(real_images)

            label = torch.full((batch_size,), real_label, device=device)
            r_loss = criterion(d_real, label)
            r_loss.backward()

            z = torch.randn(batch_size, z_size, 1, 1, device=device)
            
            fake_images = gen(z)

            label.fill_(fake_label)

            d_fake = dis(fake_images.detach())

            f_loss = criterion(d_fake, label)
            f_loss.backward()

            d_loss = r_loss + f_loss

            d_opt.step()


            #### Train the Generator ####
            g_opt.zero_grad()

            label.fill_(real_label)
            d_fake2 = dis(fake_images)

            g_loss = criterion(d_fake2, label)
            g_loss.backward()

            g_opt.step()

            if batch_i % p_every == 0:
                print ('Epoch [{:5d} / {:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'. \
                        format(e+1, epochs, d_loss, g_loss))

        train_losses.append([td_loss, tg_loss])

        if e % s_every == 0:
            d_ckpt = {
                'model_state_dict' : dis.state_dict(),
                'opt_state_dict' : d_opt.state_dict()
            }

            g_ckpt = {
                'model_state_dict' : gen.state_dict(),
                'opt_state_dict' : g_opt.state_dict()
            }

            torch.save(d_ckpt, 'd-nm-{}.pth'.format(e))
            torch.save(g_ckpt, 'g-nm-{}.pth'.format(e))

        utils.save_image(fake_images.detach(), 'fake_{}.png'.format(e), normalize=True)

    print ('[INFO] Training Completed successfully!')
