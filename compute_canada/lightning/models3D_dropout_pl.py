import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict


# ------- Creating a new architecture for 3D U-Net Generator with Spectral Normalization and Residual Blocks
class Normalconv3DBlock(nn.Module):
    def __init__(self, in_features, out_features, use_dropout=False, residual=None):
        super(Normalconv3DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(out_features),
            nn.LeakyReLU(0.2, True))
        if use_dropout:
            self.conv1.add_module('drop1', nn.Dropout(0.2))

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(out_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(out_features),
            nn.LeakyReLU(0.2, True))

        # commented because unnecessarily confusing
        # self.residual = residual
        # if self.residual is not None:
        #     self.residual_ups = spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=1))

    def forward(self, x):
        return self.conv2(self.conv1(x))
        # res = x
        # if not self.residual:
        #     return self.conv2(self.conv1(x))
        # else:
        #     return self.conv2(self.conv1(x)) + self.residual_ups(res)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(Deconv3DBlock, self).__init__()

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.ReplicationPad3d(1),
            spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3)),
            nn.InstanceNorm3d(out_features),
            nn.ReLU(True))

    def forward(self, x):
        return self.deconv(x)


class conv3DBlock(nn.Module):
    def __init__(self, in_features, out_features, use_dropout=False):
        super(conv3DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm3d(out_features),
            nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(out_features, out_features, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm3d(out_features),
            nn.LeakyReLU(0.2, True))
        if use_dropout:
            self.conv2.add_module('drop2', nn.Dropout(0.2))

    def forward(self, x):
        return self.conv2(self.conv1(x))


# ------- Creating a new architecture for 3D U-Net Generator with Spectral Normalization
class LighterUnetGenerator3D(nn.Module):

    def __init__(self, in_channels, out_channels, num_feat_maps=[32, 64, 128, 256, 512], use_dropout=False):
        super(LighterUnetGenerator3D, self).__init__()

        # Downsampling Path (using dropout only in the 3rd and 4th downsampling block)
        self.conv_block1 = conv3DBlock(in_channels, num_feat_maps[0], use_dropout)  # does not downsample in the 1st block

        # uncomment below if dropout to be included from the 2nd downsampling block itself.
        # self.conv_block2 = conv3DBlock(num_feat_maps[0], num_feat_maps[1], use_dropout=True, drop_ratio=0.2)
        self.conv_block2 = conv3DBlock(num_feat_maps[0], num_feat_maps[1], use_dropout)

        self.conv_block3 = conv3DBlock(num_feat_maps[1], num_feat_maps[2], use_dropout=True)
        self.conv_block4 = conv3DBlock(num_feat_maps[2], num_feat_maps[3], use_dropout=True)
        # adding one more layer to avoid memory issues
        self.conv_block5 = conv3DBlock(num_feat_maps[3], num_feat_maps[4], use_dropout=True)

        # Upsampling Path
        # adding one more layer to avoid memory issues
        self.deconv_block4 = Deconv3DBlock(num_feat_maps[4], num_feat_maps[3])

        self.deconv_block3 = Deconv3DBlock(num_feat_maps[3], num_feat_maps[2])
        self.deconv_block2 = Deconv3DBlock(num_feat_maps[2], num_feat_maps[1])
        self.deconv_block1 = Deconv3DBlock(num_feat_maps[1], num_feat_maps[0])
        self.deconv_block0 = Deconv3DBlock(num_feat_maps[0], num_feat_maps[0])

        # Convolutions RIGHT AFTER upsampling (using dropout (likewise) in the 3rd and 2nd upsampling)
        # adding one more layer to avoid memory issues
        self.ups_conv_block4 = Normalconv3DBlock(2 * num_feat_maps[3], num_feat_maps[3], use_dropout=True)

        self.ups_conv_block3 = Normalconv3DBlock(2 * num_feat_maps[2], num_feat_maps[2], use_dropout=True)
        self.ups_conv_block2 = Normalconv3DBlock(2 * num_feat_maps[1], num_feat_maps[1], use_dropout=True)

        self.ups_conv_block1 = Normalconv3DBlock(2 * num_feat_maps[0], num_feat_maps[0],use_dropout)
        # uncomment only if the corresponding downsampling block is uncommented
        # self.ups_conv_block1 = Normalconv3DBlock(2 * num_feat_maps[0], num_feat_maps[0], use_dropout, drop_ratio)
        self.ups_conv_block0 = Normalconv3DBlock(num_feat_maps[0], num_feat_maps[0], use_dropout)

        self.conv_1x1 = spectral_norm(nn.Conv3d(num_feat_maps[0], out_channels, kernel_size=1, stride=1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.shape)
        x1 = self.conv_block1(x)
        # print("Down Level 1: ", x1.shape)
        x2 = self.conv_block2(x1)
        # print("Down Level 2: ", x2.shape)
        x3 = self.conv_block3(x2)
        # print("Down Level 3: ", x3.shape)

        # added these lines
        x4 = self.conv_block4(x3)
        bottle_neck = self.conv_block5(x4)

        # bottle_neck = self.conv_block4(x3)            # uncomment these lines
        # print("BottleNeck: ", bottle_neck.shape)

        # added these lines
        d4 = torch.cat([self.deconv_block4(bottle_neck), x4], dim=1)
        d_ups_4 = self.ups_conv_block4(d4)
        d3 = torch.cat([self.deconv_block3(d_ups_4), x3], dim=1)

        # d3 = torch.cat([self.deconv_block3(bottle_neck), x3], dim=1)      # uncomment this
        d_ups_3 = self.ups_conv_block3(d3)
        # print("Up Level 3: ", d_ups_3.shape)
        d2 = torch.cat([self.deconv_block2(d_ups_3), x2], dim=1)
        d_ups_2 = self.ups_conv_block2(d2)
        # print("Up Level 2: ", d_ups_2.shape)
        d1 = torch.cat([self.deconv_block1(d_ups_2), x1], dim=1)
        d_ups_1 = self.ups_conv_block1(d1)
        # print("Up Level 1: ", d_ups_1.shape)

        d0 = self.deconv_block0(d_ups_1)
        d_ups_0 = self.ups_conv_block0(d0)
        # print("Up Level 0: ", d_ups_0.shape)

        fin = self.tanh(self.conv_1x1(d_ups_0))
        # print("Final Output: ", fin.shape)

        # splitting the output's ("fin") head to get two outputs now  -> For Grayscale images
        mu = (fin[:, 0]).unsqueeze(dim=1)          # this being the standard image - size: Bx1x(breadth)xHxW
        log_sigma = (fin[:, -1]).unsqueeze(dim=1)    # this being the uncertainty for input; size: Bx1xHxW

        return mu, log_sigma

# Changelog for PatchGAN discriminator:
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

        # This PatchGAN architecture has 5 layers. 46x46x46 sized patches are seen by
        # the discriminator.
        # EDIT: Instance Norm is added before every leakyReLU layer.
        model = [spectral_norm(nn.Conv3d(input_nc, 32, kernel_size=4, stride=2, padding=1)),    # from input_nc, 64
                 nn.InstanceNorm3d(32),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),    # 64, 128
                  nn.InstanceNorm3d(64),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=1, padding=1)),   # 128, 256
                  nn.InstanceNorm3d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=1)),   # 256, 512
                  nn.InstanceNorm3d(256),
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

# #################################################################################
# --------------------------- Resnet Architecture Start ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, use_dropout=False):
        super(ResidualBlock, self).__init__()
        # here we are not writing self.in_features = in_features because were are not going to use in_features in any
        # other function definition other than __init__
        conv_block = [nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv3d(in_features, in_features, 3)),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv3d(in_features, in_features, 3)),
                      nn.InstanceNorm3d(in_features)]
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# Changelog for 3D Resnet:
#   1) ReflectionPad is not there for 3D, therefore, ReplicationPad3d is used in place of that.
#   2) The default number of residual blocks is changed from 9 to 6.


class ResnetGenerator3D(nn.Module):
    # The paper mentions: "We use 6 residual blocks for 128 × 128 training images, and 9 residual blocks for
    # 256 × 256 or higher-resolution training images."
    def __init__(self, in_channels, out_channels, init_features=32, num_residual_blocks=9, use_dropout=False):
        super(ResnetGenerator3D, self).__init__()
        # Initial convolution block for the input
        model = [nn.ReplicationPad3d(3),    # nn.ReflectionPad2d(3),
                 spectral_norm(nn.Conv3d(in_channels, init_features, 7)),
                 nn.InstanceNorm3d(init_features),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = init_features
        out_features = in_features * 2

        for i in range(2):
            model += [spectral_norm(nn.Conv3d(in_features, out_features, 3, stride=2, padding=1)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            if use_dropout:
                model += [nn.Dropout(0.2)]

            in_features = out_features
            out_features = in_features * 2

        # Concatenating the Residual Blocks
        for i in range(num_residual_blocks):
            model += [ResidualBlock(in_features, use_dropout)]

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
                      spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=0)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True) ]
            if use_dropout:
                model += [nn.Dropout(0.2)]

            in_features = out_features
            out_features = in_features//2

        # the final Output layer
        model += [nn.ReplicationPad3d(3), # nn.ReflectionPad2d(3),
                  spectral_norm(nn.Conv3d(init_features, out_channels, 7)),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # "Standard forward function"
        # return self.model(x)
        # Modified for Aleatoric Uncertainty
        # splitting the output's ("out") head to get two outputs now  -> For Grayscale images
        out = self.model(x)
        mu = (out[:, 0]).unsqueeze(dim=1)     # this being the standard RGB image - size: BxCxHxW
        log_sigma = (out[:, -1]).unsqueeze(dim=1)   # this being the uncertainty for the particular input - size: BxHxW
        return mu, log_sigma


# ------------------------------ ResUnet Architecture Start -------------------------------
class DownConv3D(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownConv3D, self).__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm3d(out_features),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.conv(x)


class LighterResUnetGenerator3D(nn.Module):

    def __init__(self, in_channels, out_channels, num_feat_maps=[32, 64, 128, 256], residual='conv'):
        super(LighterResUnetGenerator3D, self).__init__()

        # Downsampling Path
        self.conv_block1 = Normalconv3DBlock(in_channels, num_feat_maps[0], residual=residual)
        self.conv_block1_down = DownConv3D(num_feat_maps[0], num_feat_maps[0])
        self.conv_block2 = Normalconv3DBlock(num_feat_maps[0], num_feat_maps[1], residual=residual)
        self.conv_block2_down = DownConv3D(num_feat_maps[1], num_feat_maps[1])
        self.conv_block3 = Normalconv3DBlock(num_feat_maps[1], num_feat_maps[2], residual=residual)
        self.conv_block3_down = DownConv3D(num_feat_maps[2], num_feat_maps[2])
        self.conv_block4 = Normalconv3DBlock(num_feat_maps[2], num_feat_maps[3], residual=residual)
        self.conv_block4_down = DownConv3D(num_feat_maps[3], num_feat_maps[3])

        # Upsampling Path
        self.deconv_block3 = Deconv3DBlock(num_feat_maps[3], num_feat_maps[2])
        self.deconv_block2 = Deconv3DBlock(num_feat_maps[2], num_feat_maps[1])
        self.deconv_block1 = Deconv3DBlock(num_feat_maps[1], num_feat_maps[0])

        self.deconv_block0 = Deconv3DBlock(num_feat_maps[0], num_feat_maps[0])

        # Convolutions RIGHT AFTER upsampling
        self.ups_conv_block3 = Normalconv3DBlock(2 * num_feat_maps[2], num_feat_maps[2], residual=residual)
        self.ups_conv_block2 = Normalconv3DBlock(2 * num_feat_maps[1], num_feat_maps[1], residual=residual)
        self.ups_conv_block1 = Normalconv3DBlock(2 * num_feat_maps[0], num_feat_maps[0], residual=residual)

        self.ups_conv_block0 = Normalconv3DBlock(num_feat_maps[0], num_feat_maps[0], residual=residual)

        self.conv_1x1 = spectral_norm(nn.Conv3d(num_feat_maps[0], out_channels, kernel_size=1, stride=1))

        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.shape)
        x1 = self.conv_block1_down(self.conv_block1(x))
        # print("Down Level 1: ", x1.shape)
        x2 = self.conv_block2_down(self.conv_block2(x1))
        # print("Down Level 2: ", x2.shape)
        x3 = self.conv_block3_down(self.conv_block3(x2))
        # print("Down Level 3: ", x3.shape)
        bottle_neck = self.conv_block4_down(self.conv_block4(x3))
        # print("BottleNeck: ", bottle_neck.shape)

        d3 = torch.cat([self.deconv_block3(bottle_neck), x3], dim=1)
        d_ups_3 = self.ups_conv_block3(d3)
        # print("Up Level 3: ", d_ups_3.shape)
        d2 = torch.cat([self.deconv_block2(d_ups_3), x2], dim=1)
        d_ups_2 = self.ups_conv_block2(d2)
        # print("Up Level 2: ", d_ups_2.shape)
        d1 = torch.cat([self.deconv_block1(d_ups_2), x1], dim=1)
        d_ups_1 = self.ups_conv_block1(d1)
        # print("Up Level 1: ", d_ups_1.shape)

        d0 = self.deconv_block0(d_ups_1)
        d_ups_0 = self.ups_conv_block0(d0)
        # print("Up Level 0: ", d_ups_0.shape)

        fin = self.tanh(self.conv_1x1(d_ups_0))
        # print("Final Output: ", fin.shape)
        return fin


