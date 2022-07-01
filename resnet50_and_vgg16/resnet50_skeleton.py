import torch.nn as nn

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return model


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride, padding):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return model

###########################################################################
# Question 1 : Implement the "bottle neck building block" part.
# Hint : Think about difference between downsample True and False. How we make the difference by code?
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            # the output image's size is half of the input image size.
            self.layer = nn.Sequential(
                ##########################################
                ############## fill in here (20 points)
                # Hint : use these functions (conv1x1, conv3x3)
                #########################################
                conv1x1(in_channels,middle_channels, 2, 0), # number of channels: in_channels -> middle_channels, size: be halved
                conv3x3(middle_channels,middle_channels, 1, 1), # number of channels & size: no changed
                conv1x1(middle_channels, out_channels, 1, 0) # number of channels: middle_channels -> out_channels, size: no changed
            )
            # the input image's size is halved.
            self.downsize = conv1x1(in_channels, out_channels, 2, 0)

        else:
            # the output image's size is not changed.
            self.layer = nn.Sequential(
                ##########################################
                ############# fill in here (20 points)
                #########################################
                conv1x1(in_channels, middle_channels, 1, 0),  # number of channels: in_channels -> middle_channels
                conv3x3(middle_channels, middle_channels, 1, 1),
                conv1x1(middle_channels, out_channels, 1, 0)  # number of channels: middle_channels -> out_channels
            )
            # make number of input image's channel = number of output image's channel
            self.make_equal_channel = conv1x1(in_channels, out_channels, 1, 0) # number of channels: in_channels -> out_channels

    def forward(self, x):
        if self.downsample:
            out = self.layer(x) # output image
            x = self.downsize(x) # input image
            return out + x # output image + input image
        else:
            out = self.layer(x) # output image
            if x.size() is not out.size():
                x = self.make_equal_channel(x)
            return out + x # output image + input image
###########################################################################



###########################################################################
# Question 2 : Implement the "class, ResNet50_layer4" part.
# Understand ResNet architecture and fill in the blanks below. (25 points)
# (blank : #blank#, 1 points per blank )
# Implement the code.
class ResNet50_layer4(nn.Module):
    def __init__(self, num_classes= 10): # Hint : How many classes in Cifar-10 dataset?
        super(ResNet50_layer4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), # input channel sizes:3, output channel sizes:64, kernel size:7, stride:2, padding:3
                # Hint : Through this conv-layer, the input image size is halved.
                #        Consider stride, kernel size, padding and input & output channel sizes.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0)  # input image size is halved.(kernel size:3, stride:2, padding:0)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, 256, False), # number of channels: 64->64->64->256, image size: no change
            ResidualBlock(256, 64, 256, False), # number of channels: 256->64->64->256, image size: no change
            ResidualBlock(256, 64, 256, True) # number of channels: 256->64->64->256, image size: be halved
        )
        self.layer3 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ####### you can refer to the 'layer2' above
            #########################################
            ResidualBlock(256, 128, 512, False), # number of channels: 256->128->128->512, image size: no change
            ResidualBlock(512, 128, 512, False), # number of channels: 512->128->128->512, image size: no change
            ResidualBlock(512, 128, 512, False),  # number of channels: 512->128->128->512, image size: no change
            ResidualBlock(512, 128, 512, True)  # number of channels: 512->128->128->512, image size: behalved
        )
        self.layer4 = nn.Sequential(
            ##########################################
            ############# fill in here (20 points)
            ####### you can refer to the 'layer2' above
            #########################################
            ResidualBlock(512, 256, 1024, False),  # number of channels: 512->256->256->1024, image size: no change
            ResidualBlock(1024, 256, 1024, False),  # number of channels: 1024->256->256->1024, image size: no change
            ResidualBlock(1024, 256, 1024, False),  # number of channels: 1024->256->256->1024, image size: no change
            ResidualBlock(1024, 256, 1024, False),  # number of channels: 1024->256->256->1024, image size: no change
            ResidualBlock(1024, 256, 1024, False),  # number of channels: 1024->256->256->1024, image size: no change
            ResidualBlock(1024, 256, 1024, False)  # number of channels: 1024->256->256->1024, image size: no change
        )

        self.fc = nn.Linear(1024, num_classes) # Hint : Think about the reason why fc layer is needed
        self.avgpool = nn.AvgPool2d(2, 1) # image size: 2x2 -> 1x1 (kernel size:2, stride:1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out
###########################################################################