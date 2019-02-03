class Discriminator(nn.Module):
    
    def __init__(self, c1_channels=128, c2_channels=256, c3_channels=512):
        '''
        The constructor method for the Discriminator class
        
        Arguments:
        - c1_channels : the number of output channels from the
                        first Convolutional Layer [Default - 128]
                        
        - c2_channels : the number of output channels from the
                        second Convolutional Layer [Default - 256]
                        
        - c3_channels : the number of output channels from the
                        third Convolutional Layer [Default - 512]
                        
        '''
        super().__init__()
        
        # Define the class variables
        self.c1_channels = c1_channels
        self.c2_channels = c2_channels
        self.c3_channels = c3_channels
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.c1_channels,
                               kernel_size=3,
                               stride=2)
        
        self.bnorm1 = nn.BatchNorm2d(out_features=self.c1_channels)
        
        self.conv2 = nn.Conv2d(in_channels=self.c1_channels,
                               out_channels=self.c2_channels,
                               kernel_size=3,
                               stride=2)
        
        self.bnorm2 = nn.BatchNorm2d(out_features=self.c2_channels)
        
        self.conv3 = nn.Conv2d(in_channels=self.c2_channels,
                               out_channels=self.c3_channels,
                               kernel_size=3,
                               stride=2)
        
        self.bnorm3 = nn.BatchNorm2d(out_features=self.c3_channels)
        
        self.conv4 = nn.Conv2d(in_channels=self.c3_channels,
                               out_channels=self.c4_channels,
                               kernel_size=3,
                               stride=2)
        
        self.lrelu = nn.LeakyReLU(negetive_slope=0.2)
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
        
        x = self.bnorm1(self.lrelu(self.conv1(img)))
        x = self.bnorm1(self.lrelu(self.conv2(img)))
        x = self.bnorm1(self.lrelu(self.conv3(img)))
        out = self.sigmoid(self.conv4(img))
        
        return out
