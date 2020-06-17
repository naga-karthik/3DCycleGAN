import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict

# ------- Creating a new architecture for 3D U-Net Generator with Spectral Normalization and Residual Blocks
class ModifiedResUnetGenerator3D(nn.Module):

    def __init__(self, in_channels, out_channels, init_featMaps=32):
        super(ModifiedResUnetGenerator3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_featMaps = init_featMaps

        self.lrelu = nn.LeakyReLU(0.2, True)    # to be used in the downsampling/context path
        self.relu = nn.ReLU(True)       # to be used in the upsampling/localization path
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.tanh = nn.Tanh()

        # Level 1 context pathway
        self.conv_c11 = spectral_norm(nn.Conv3d(self.in_channels, self.init_featMaps, kernel_size=3, stride=1, padding=1))

        self.conv_c1 = spectral_norm(nn.Conv3d(self.init_featMaps, self.init_featMaps, kernel_size=3, stride=1, padding=1))
        self.lrelu_conv_c1 = self.lrelu_conv_block(self.init_featMaps, self.init_featMaps)
        self.norm_c1 = nn.InstanceNorm3d(init_featMaps)

        # Level 2 context pathway
        self.conv_c2 = spectral_norm(nn.Conv3d(self.init_featMaps, self.init_featMaps*2, kernel_size=3, stride=2, padding=1))
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv_block(self.init_featMaps*2, self.init_featMaps*2)
        self.norm_c2 = nn.InstanceNorm3d(self.init_featMaps*2)

        # Level 3 context pathway
        self.conv_c3 = spectral_norm(nn.Conv3d(self.init_featMaps*2, self.init_featMaps*4, kernel_size=3, stride=2, padding=1))
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv_block(self.init_featMaps*4, self.init_featMaps*4)
        self.norm_c3 = nn.InstanceNorm3d(self.init_featMaps*4)

        # Level 4 context pathway, level 0 localization pathway (bottleneck)
        self.conv_c4 = spectral_norm(nn.Conv3d(self.init_featMaps*4, self.init_featMaps*8, kernel_size=3, stride=2, padding=1))
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv_block(self.init_featMaps*8, self.init_featMaps*8)
        self.norm_c4 = nn.InstanceNorm3d(self.init_featMaps*8)

        # self.norm_relu_ups_l0 = self.norm_relu_upsample_conv_norm_relu_block(self.init_featMaps*8, self.init_featMaps*4)
        # self.conv_l0 = nn.Conv3d(self.init_featMaps*4, self.init_featMaps*4, kernel_size=1, stride=1, padding=0)
        # self.norm_l0 = nn.InstanceNorm3d(self.init_featMaps*4)

        # Level 5 context pathway, level 0 localization pathway (bottleneck)
        self.conv_c5 = spectral_norm(nn.Conv3d(self.init_featMaps*8, self.init_featMaps*16, kernel_size=3, stride=2, padding=1))
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv_block(self.init_featMaps*16, self.init_featMaps*16)

        self.norm_relu_ups_l0 = self.norm_relu_upsample_conv_norm_relu_block(self.init_featMaps*16,self.init_featMaps*8)
        self.conv_l0 = spectral_norm(nn.Conv3d(self.init_featMaps * 8, self.init_featMaps * 8, kernel_size=1, stride=1, padding=0))
        self.norm_l0 = nn.InstanceNorm3d(self.init_featMaps * 8)

        # Level 1 localization pathway
        self.conv_norm_relu_l1 = self.conv_norm_relu_block(self.init_featMaps*16, self.init_featMaps*16)
        self.conv_l1 = spectral_norm(nn.Conv3d(self.init_featMaps*16, self.init_featMaps*8, kernel_size=1, stride=1, padding=0))
        self.norm_relu_ups_l1 = self.norm_relu_upsample_conv_norm_relu_block(self.init_featMaps*8, self.init_featMaps*4)

        # Level 2 localization pathway
        self.conv_norm_relu_l2 = self.conv_norm_relu_block(self.init_featMaps*8, self.init_featMaps*8)
        self.conv_l2 = spectral_norm(nn.Conv3d(self.init_featMaps*8, self.init_featMaps*4, kernel_size=1, stride=1, padding=0))
        self.norm_relu_ups_l2 = self.norm_relu_upsample_conv_norm_relu_block(self.init_featMaps*4, self.init_featMaps*2)

        # Level 3 localization pathway
        self.conv_norm_relu_l3 = self.conv_norm_relu_block(self.init_featMaps*4, self.init_featMaps*4)
        self.conv_l3 = spectral_norm(nn.Conv3d(self.init_featMaps*4, self.init_featMaps*2, kernel_size=1, stride=1, padding=0))
        self.norm_relu_ups_l3 = self.norm_relu_upsample_conv_norm_relu_block(self.init_featMaps*2, self.init_featMaps)

        # Level 4 localization pathway
        self.conv_norm_relu_l4 = self.conv_norm_relu_block(self.init_featMaps*2, self.init_featMaps*2)
        self.conv_l4 = spectral_norm(nn.Conv3d(self.init_featMaps*2, self.out_channels, kernel_size=1, stride=1, padding=0))

        self.ds2_conv1x1 = spectral_norm(nn.Conv3d(self.init_featMaps*8, self.out_channels, kernel_size=1, padding=0, stride=1))
        self.ds3_conv1x1 = spectral_norm(nn.Conv3d(self.init_featMaps*4, self.out_channels, kernel_size=1, padding=0, stride=1))

    def lrelu_conv_block(self, in_features, out_features):
        return nn.Sequential(self.lrelu,
                             spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)))

    def norm_lrelu_conv_block(self, in_features, out_features):
        return nn.Sequential(nn.InstanceNorm3d(in_features),
                             self.lrelu,
                             spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)))

    def conv_norm_relu_block(self, in_features, out_features):
        return nn.Sequential(spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
                             nn.InstanceNorm3d(in_features),
                             self.relu)

    def norm_relu_upsample_conv_norm_relu_block(self, in_features, out_features):
        return nn.Sequential(nn.InstanceNorm3d(in_features),
                             self.relu,
                             self.upscale,
                             spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
                             nn.InstanceNorm3d(in_features),
                             self.relu)

    def forward(self, x):
        # Level 1 pathway
        out = self.conv_c11(x)
        residual_1 = out
        out = self.lrelu_conv_c1(self.conv_c1(out))     # skipped dropout and leaky relu
        # print("Level 1 context before residual: ", out.shape)
        out += residual_1   # simple residual summation
        context_1 = out
        out = self.lrelu(self.norm_c1(out))
        # print("Level 1 context after residual: ", out.shape)

        # Level 2 pathway
        out = self.conv_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        # print("Level 2 context before residual: ", out.shape)
        out += residual_2
        out = self.lrelu(self.norm_c2(out))
        context_2 = out
        # print("Level 2 context after residual: ", out.shape)

        # Level 3 pathway
        out = self.conv_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        # print("Level 3 context before residual: ", out.shape)
        out += residual_3
        out = self.lrelu(self.norm_c3(out))
        context_3= out
        # print("Level 3 context after residual: ", out.shape)

        # Level 4 pathway
        out = self.conv_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        # print("Level 4 context before residual: ", out.shape)
        out += residual_4
        out = self.lrelu(self.norm_c4(out))
        context_4 = out
        # print("Level 4 context after residual: ", out.shape)

        # Level 5 pathway
        out = self.conv_c5(out)
        residual_5 = out
        # print("Level 5 context before residual: ", out.shape)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        # print("Level 5 context after residual: ", out.shape)

        # Localization level 0
        out = self.norm_relu_ups_l0(out)
        out = self.relu(self.norm_l0(self.conv_l0(out)))
        # print("BottleNeck: ", out.shape)

        # Level 1 localization pathway
        # print("Level 1 local before concat: ", out.shape)
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_relu_l1(out)
        # ds1 = out
        out = self.norm_relu_ups_l1(self.conv_l1(out))
        # print("Level 1 local after concat: ", out.shape)

        # Level 2 localization pathway
        # print("Level 2 local before concat: ", out.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_relu_l2(out)
        ds2 = out
        out = self.norm_relu_ups_l2(self.conv_l2(out))
        # print("Level 2 local after concat: ", out.shape)

        # Level 3 localization pathway
        # print("Level 3 local before concat: ", out.shape)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_relu_l3(out)
        ds3 = out
        out = self.norm_relu_ups_l3(self.conv_l3(out))
         # print("Level 3 local after concat: ", out.shape)

        # Level 4 localization pathway
        # print("Level 4 local before concat: ", out.shape)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_relu_l4(out)
        # print("Level 4 local before concat: ", out.shape)
        out_pred = self.conv_l4(out)

        # adding the residuals in the localization pathway
        ds1_ds2_ups = self.upscale(self.ds2_conv1x1(ds2))
        ds3_conv = self.ds3_conv1x1(ds3)
        ds2_ds3_ups = self.upscale(ds1_ds2_ups + ds3_conv)
        out = out_pred + ds2_ds3_ups
        # print("final output shape: ", out.shape)
        return self.tanh(out)


# -------- Adding the option of the possibility of using a Unet Generator --------
class UnetDownBlockUpdated(nn.Module):

    def __init__(self, in_channels, features, name):
        super(UnetDownBlockUpdated, self).__init__()
        self.unet_downblock = nn.Sequential(
            OrderedDict(
                [(name + 'conv1',
                  spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=(2, 4, 4),
                                          stride=2, padding=1, bias=False))),
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
                [((name + 'conv1'), spectral_norm(nn.Conv3d(in_channels=features*2, out_channels=features,
                                                            kernel_size=1))),
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

# Changelog for updated 3D Unet architecture:
#   1) Kernel size is changed to (2,4,4) so that the volume is reduced to a voxel of size 2x2x1 at the bottleNeck.
#   2) Even for ConvTranspose3d, the kernel size is changed from 3 across all dimensions to (2,4,4) for each
#      dimension (z,x,y).
#   3) For concatenating with correct dimensions, the "output_padding" argument is tweaked accordingly.
#   4) The initial number of feature maps has also been increased from 32 to 64.


class UnetGenerator3DUpdated(nn.Module):

    def __init__(self, in_channels, out_channels, init_featMaps=64):
        super(UnetGenerator3DUpdated, self).__init__()
        features = init_featMaps
        # CONTRACTIVE PATH BEGIN According to the given input size of 300x300, when the bottleneck layer is reached,
        # the size of the image is reduced to a single digitxdigit
        self.encoder1 = UnetDownBlockUpdated(in_channels, features, name='enc1')  # 1, 64
        self.encoder2 = UnetDownBlockUpdated(features, features * 2, name='enc2')   # 64, 128
        self.encoder3 = UnetDownBlockUpdated(features * 2, features * 4, name='enc3')  # 128, 256
        self.encoder4 = UnetDownBlockUpdated(features * 4, features * 8, name='enc4')  # 256, 512
        self.encoder5 = UnetDownBlockUpdated(features * 8, features * 16, name='enc5')  # 512, 1024
        self.encoder6 = UnetDownBlockUpdated(features * 16, features * 16, name='enc6')  # 1024, 1024

        # this marks the end of the downhill (downsample) path of the U-shaped network
        self.bottleNeck = UnetDownBlockUpdated(features * 16, features * 16, name='bottleNeck')  # 1024, 1024

        # EXPANSIVE PATH BEGIN
        self.upconv6 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 16,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(0, 0, 0)))
        self.decoder6 = UnetUpBlock(features * 16, name='dec4')
        self.upconv5 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 16,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(1, 0, 0)))
        self.decoder5 = UnetUpBlock(features * 16, name='dec5')
        self.upconv4 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 16, out_channels=features * 8,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(0, 0, 0)))
        self.decoder4 = UnetUpBlock(features * 8, name='dec4')  # it is here that outputs from contractive path will be concatenated.
        self.upconv3 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 8, out_channels=features * 4,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(1, 0, 0)))
        self.decoder3 = UnetUpBlock(features * 4, name='dec3')
        self.upconv2 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 4, out_channels=features * 2,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(1, 0, 0)))
        self.decoder2 = UnetUpBlock(features * 2, name='dec2')
        self.upconv1 = spectral_norm(nn.ConvTranspose3d(in_channels=features * 2, out_channels=features,
                                                        kernel_size=(2, 4, 4), stride=2, padding=1,
                                                        output_padding=(1, 0, 0)))
        self.decoder1 = UnetUpBlock(features, name='dec1')

        self.conv = spectral_norm(nn.ConvTranspose3d(in_channels=features, out_channels=out_channels,
                                                     kernel_size=(2, 4, 4), stride=2, padding=1,
                                                     output_padding=(0,0,0)))
        # self.conv = spectral_norm(nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1))

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
        enc6 = self.encoder6(enc5)
        # print(enc6.shape)

        # TRANSITION STATE: CONTRACTIVE END, EXPANSIVE BEGIN
        bottleNeck = self.bottleNeck(enc6)
        # bottleNeck = self.bottleNeck(enc5)
        # print("bottleNeck: ", bottleNeck.shape)

        # EXPANSIVE PATH
        dec6 = self.upconv6(bottleNeck)
        # print(dec6.shape)
        dec6 = torch.cat((dec6, enc6), dim=1)  # column wise concatenation
        dec6 = self.decoder6(dec6)
        # print(dec6.shape)

        dec5 = self.upconv5(dec6)
        # dec5 = self.upconv5(bottleNeck)
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
#   1) It now consists of 6 layers instead of 5. * this was before, it has now been reverted back to 5 layers
#   2) The architecture is such that 46x46x46 patches are seen by the discriminator.
#   3) In one of the questions on Github's issues, it was noted that for calculating the patch size seen, the padding
#      is not taken into account. (o/p size - 1)*stride + kernel size = i/p size. Padding is just seen as an indicator
#      for localizing the boundary.
#   4) Reducing the capacity of the discriminator by decreasing the number of filters by 2 for each layer

class PatchGANDiscriminatorwithSpectralNorm(nn.Module):

    # The paper mentions: "For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a  4 × 4
    # Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a
    # convolution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky
    # ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512"

    def __init__(self, input_nc):
        super(PatchGANDiscriminatorwithSpectralNorm, self).__init__()

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
        model = [spectral_norm(nn.Conv3d(input_nc, 32, 4, stride=2, padding=1)),    # from input_nc, 64
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),    # 64, 128
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=1, padding=1)),   # 128, 256
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=1)),   # 256, 512
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(256, 1, kernel_size=4, padding=1))]       # 512, 1

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # since a fully connected layer is not used as the output layer, an average pooling is used as a replacement
        # it is also flattened and sent as the output
        # x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
        # print(x.size(), x.size()[2:])
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0])
        return x
