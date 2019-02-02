import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):

    def __init__(self, zin_channels):

        super().__init__()

        # Define the class variables
        self.zin_channels = zin_channels

        self.convt_1 = nn.ConvTranspose2d(in_channels=self.zin_channels,
                                          out_channels=1024,
                                          kernel_size=1,
                                          stride=2)

        self.convt_2 = nn.ConvTranspose2d(in_channels=1024,
                                          out_channels=512,
                                          kernel_size=5,
                                          stride=2)


        self.convt_3 = nn.ConvTranspose2d(in_channels=512,
                                          out_channels=256,
                                          kernel_size=5,
                                          stride=2)

        self.convt_4 = nn.ConvTranspose2d(in_channels=256,
                                          out_channels=128,
                                          kernel_size=5,
                                          stride=2)

        self.convt_5 = nn.ConvTranspose2d(in_channels=128,
                                          out_channels=3,
                                          kernel_size=5,
                                          stride=2)
    
    def forward(self, z):
        # TODO: reshape z

        x = self.convt_1(z)
        x = self.convt_2(x)
        x = self.convt_3(x)
        x = self.convt_4(x)

        out = self.convt_5(x)

        return out
