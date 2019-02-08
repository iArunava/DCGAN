# Discriminator

# Probably a VGG16 or VGG19 for Simple Image Classification pretrained on ImageNet

class Discriminator(nn.Module):
    
    def __init__(self, c1_channels=64, c2_channels=128, c3_channels=256,
                 c4_channels=512, i_channels_in_2=True):
        '''
        The constructor method for the Discriminator class
        
        Arguments:
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
        
        self.fc1 = nn.Linear(8192, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        
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
        x = self.bnorm2(self.lrelu(self.conv2(x)))
        x = self.bnorm3(self.lrelu(self.conv3(x)))
        x = self.bnorm4(self.lrelu(self.conv4(x)))
        
        #print (x.size())
        
        x = x.view(batch_size, -1)
        #print (x.shape)
        
        #print (x.size())
        
        out = self.fc1(x)
        
        return out
