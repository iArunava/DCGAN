####################################################################
# Reproduced from the paper Unsupervised Representation Learning   #
# with Deep Convolutional Generative Adversarial Networks - DCGAN  #
# 								   #
# Feel free to reuse and modify all the code in this file with     #
# proper linkage directing back to this repository.                #
#							  	   #
# Author: @iArunava						   #
#								   #
####################################################################

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ct1_channels=512, ct2_channels=256,
                 ct3_channels=128, ct4_channels=64, d_channels_in_2=False, z_size=100):
        
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
