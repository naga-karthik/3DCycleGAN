import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # here we are not writing self.in_features = in_features because were are not going to use in_features in any
        # other function definition other than __init__
        conv_block = [nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      nn.Conv3d(in_features, in_features, 3),
                      nn.InstanceNorm3d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# Changelog for 3D Resnet:
#   1) ReflectionPad is not there for 3D, therefore, ReplicationPad3d is used in place of that.
#   2) The default number of residual blocks is changed from 9 to 6.

class ResnetGenerator3D(nn.Module):

    # The paper mentions: "We use 6 residual blocks for 128 × 128 training images, and 9 residual blocks for
    # 256 × 256 or higher-resolution training images."

    def __init__(self, input_nc, output_nc, num_residual_blocks=6):
        super(ResnetGenerator3D, self).__init__()

        # Initial convolution block for the input
        model = [nn.ReplicationPad3d(3),    # nn.ReflectionPad2d(3),
                 nn.Conv3d(input_nc, 64, 7),
                 nn.InstanceNorm3d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        for i in range(2):
            model += [nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Concatenating the Residual Blocks
        for i in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for i in range(2):
            # model += [nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            #           nn.InstanceNorm3d(out_features),
            #           nn.ReLU(inplace=True)]
            # in_features = out_features
            # out_features = in_features // 2

            # Trying the resize-convolution method mentioned in the Distill article to avoid checkerboard artifacts
            # while upsampling using ConvTranspose2d
            model += [nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReplicationPad3d(1),   # nn.ReflectionPad2d(1),
                      nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=0),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True) ]
            in_features=out_features
            out_features=in_features//2

        # the final Output layer
        model += [nn.ReplicationPad3d(3), # nn.ReflectionPad2d(3),
                  nn.Conv3d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        "Standard forward function"
        return self.model(x)


# -------- Adding the option of the possibility of using a Unet Generator --------
class UnetDownBlock(nn.Module):

    def __init__(self, in_channels, features, name):
        super(UnetDownBlock, self).__init__()
        self.unet_downblock = nn.Sequential(
            OrderedDict(
                [(name + 'conv1',
                  nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, stride=2, padding=1, bias=False)),
                 (name + 'norm1', nn.InstanceNorm3d(num_features=features)),
                 (name + 'relu1', nn.LeakyReLU(0.2, inplace=True)) ]
                 # (name + 'conv2',
                 #  nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                 # (name + 'norm2', nn.BatchNorm2d(num_features=features)),
                 # (name + 'relu2', nn.ReLU(inplace=True))]
            )
        )
        # self.unet_block = nn.Sequential(*unet_block)

    def forward(self, x):
        return self.unet_downblock(x)


class UnetUpBlock(nn.Module):

    def __init__(self, features, name):
        super(UnetUpBlock, self).__init__()
        self.unet_upblock = nn.Sequential(
            OrderedDict(
                [((name + 'conv1'), nn.Conv3d(in_channels=features*2, out_channels=features, kernel_size=1)),
                 (name + 'norm1', nn.InstanceNorm3d(num_features=features)),
                 (name + 'relu1', nn.ReLU(inplace=True)) ]
                 # (name + 'conv2',
                 #  nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)),
                 # (name + 'norm2', nn.BatchNorm2d(num_features=features)),
                 # (name + 'relu2', nn.ReLU(inplace=True))]
            )
        )
        # self.unet_block = nn.Sequential(*unet_block)

    def forward(self, x):
        return self.unet_upblock(x)

# Changelog for 3D Unet architecture:
#   1) Kernel size is changed from 4 to 3 in UnetDownBlock for solving some size issues.
#   2) An encoder-decoder (enc6-dec6) pair is removed.
#   3) Even for ConvTranspose3d, the kernel size is changed from 4 to 3.
#   4) For concatenating with correct dimensions, the "output_padding" argument is tweaked accordingly.

class UnetGenerator3D(nn.Module):

    def __init__(self, in_channels, out_channels, init_featMaps=32):
        super(UnetGenerator3D, self).__init__()
        features = init_featMaps
        # CONTRACTIVE PATH BEGIN According to the given input size of 300x300, when the bottleneck layer is reached,
        # the size of the image is reduced to a single digitxdigit
        self.encoder1 = UnetDownBlock(in_channels, features, name='enc1')  # 1, 32
        self.encoder2 = UnetDownBlock(features, features * 2, name='enc2') # 32, 64
        self.encoder3 = UnetDownBlock(features * 2, features * 4, name='enc3')  # 64, 128
        self.encoder4 = UnetDownBlock(features * 4, features * 8, name='enc4')  # 128, 256
        self.encoder5 = UnetDownBlock(features * 8, features * 16, name='enc5')  # 256, 512
        # self.encoder6 = UnetDownBlock(features * 16, features * 16, name='enc6')  # 512, 512

        # this marks the end of the downhill (downsample) path of the U-shaped network
        self.bottleNeck = UnetDownBlock(features * 16, features * 16, name='bottleNeck')  # 512, 512

        # EXPANSIVE PATH BEGIN
        # self.upconv6 = nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        # self.decoder6 = UnetUpBlock(features * 16, name='dec4')
        self.upconv5 = nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 16, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        self.decoder5 = UnetUpBlock(features * 16, name='dec5')
        self.upconv4 = nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 8, kernel_size=3, stride=2, padding=1, output_padding=(0,1,1))
        self.decoder4 = UnetUpBlock(features * 8, name='dec4')  # it is here that outputs from contractive path will be concatenated.
        self.upconv3 = nn.ConvTranspose3d(in_channels=features * 8, out_channels=features * 4, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        self.decoder3 = UnetUpBlock(features * 4, name='dec3')
        self.upconv2 = nn.ConvTranspose3d(in_channels=features * 4, out_channels=features * 2, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        self.decoder2 = UnetUpBlock(features * 2, name='dec2')
        self.upconv1 = nn.ConvTranspose3d(in_channels=features * 2, out_channels=features, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        self.decoder1 = UnetUpBlock(features, name='dec1')

        self.conv = nn.ConvTranspose3d(in_channels=features, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=(1,1,1))
        # self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # CONTRACTIVE PATH
        enc1 = self.encoder1(x)
        # print(enc1.shape)
        enc2 = self.encoder2(enc1)
        # print(enc2.shape)
        enc3 = self.encoder3(enc2)
        # print(enc3.shape)
        enc4 = self.encoder4(enc3)
        # print(enc4.shape)
        enc5 = self.encoder5(enc4)
        # print(enc5.shape)
        # enc6 = self.encoder6(enc5)
        # print(enc6.shape)

        # TRANSITION STATE: CONTRACTIVE END, EXPANSIVE BEGIN
        # bottleNeck = self.bottleNeck(enc6)
        bottleNeck = self.bottleNeck(enc5)
        # print("bottleNeck: ", bottleNeck.shape)

        # EXPANSIVE PATH
        # dec6 = self.upconv6(bottleNeck)
        # dec6 = torch.cat((dec6, enc6), dim=1)  # column wise concatenation
        # dec6 = self.decoder6(dec6)
        # # print(dec6.shape)

        # dec5 = self.upconv5(dec6)
        dec5 = self.upconv5(bottleNeck)
        dec5 = torch.cat((dec5, enc5), dim=1)  # column wise concatenation
        dec5 = self.decoder5(dec5)
        # print("D5: ", dec5.shape)

        dec4 = self.upconv4(dec5)
        # print(dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)  # column wise concatenation
        dec4 = self.decoder4(dec4)
        # print("D4: ", dec4.shape)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        # print(dec3.shape)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        # print(dec2.shape)

        dec1 = self.upconv1(dec2)
        # print(dec1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # print("D1: ", dec1.shape)

        return torch.tanh(self.conv(dec1))


# Changelog for PatchGAN discriminator:
#   1) It now consists of 6 layers instead of 5.
#   2) The architecture is such that 48x48x48 patches are seen by the discriminator.
#   3) padding is removed in all layers, except the first.

class PatchGANDiscriminator(nn.Module):

    # The paper mentions: "For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a  4 × 4
    # Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a
    # convolution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky
    # ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512"

    def __init__(self, input_nc):
        super(PatchGANDiscriminator, self).__init__()

        # # this is just a classifier, so the architecture is just like a normal image classifier CNN
        # model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
        #          nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #           nn.InstanceNorm2d(128),
        #           nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        #           nn.InstanceNorm2d(256),
        #           nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        #           nn.InstanceNorm2d(512),
        #           nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]
        # # instead of a fully connected layer, the final layer is a convolutional layer (greatly reduces the total
        # # number of parameters of the network)

        # This PatchGAN architecture has 6 layers (1 more than the original). 46x46x46 sized patches are seen by
        # the discriminator.
        model = [nn.Conv3d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm3d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm3d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv3d(256, 512, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm3d(512),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv3d(512, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # since a fully connected layer is not used as the output layer, an average pooling is used as a replacement
        # it is also flattened and sent as the output
        # x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
        # print(x.size(), x.size()[2:])
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0])
        return x
