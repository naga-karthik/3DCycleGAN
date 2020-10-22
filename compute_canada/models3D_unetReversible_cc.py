import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import revtorch.revtorch

NUM_CHANNELS = [20, 40, 80, 160]
# NUM_CHANNELS = [32, 64, 128, 256]
# NUM_CHANNELS = [16, 32, 64, 128]


class ResidualInner(nn.Module):
    '''
    Class for creating the constituent NNs (i.e. Fs and Gs) in each reversible block
    '''
    def __init__(self, channels):
        super(ResidualInner, self).__init__()
        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.conv_block(x)


def makeReversibleSequence(channels):
    inner_channels = channels//2
    f_block = ResidualInner(inner_channels)
    g_block = ResidualInner(inner_channels)
    return revtorch.ReversibleBlock(f_block, g_block)


def makeReversibleComponent(channels, num_blocks):
    modules = []
    # for number of reversible blocks in a reversible sequence. Remember that theoretically there can be infinitely
    # many such blocks in a sequence
    for i in range(num_blocks):
        modules.append(makeReversibleSequence(channels))
    return revtorch.ReversibleSequence(nn.ModuleList(modules))


def getChannelsAtIndex(index, channels):
    if index < 0: index=0
    if index >= len(channels): index = len(channels)-1
    return channels[index]


class EncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, downsample=True):
        super(EncoderModule, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
            self.conv_down = nn.Sequential(
                spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
        self.reversible_blocks = makeReversibleComponent(out_channels, num_blocks=depth)

    def forward(self, x):
        print("Shape after entering the Encoder Module: ", x.shape)
        if self.downsample:
            x = self.conv_down(self.conv(x))
            print("Shape after downsampling: ", x.shape)
        x = self.reversible_blocks(x)
        # print("Shape after passing it through the Reversible block: ", x.shape)
        return x


class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, depth, upsample=True, firstUpsamplingLayer=False):
        super(DecoderModule, self).__init__()
        self.upsample = upsample
        self.firstUpsamplingLayer = firstUpsamplingLayer
        if self.upsample:
            self.deconv_ups = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReplicationPad3d(1),
                spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3)),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(True)
            )
            self.conv_ups = nn.Sequential(
                spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
            self.conv_first_ups = nn.Sequential(
                spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=(4,3,3), padding=1)),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, True)
            )
        self.reversible_blocks = makeReversibleComponent(in_channels, num_blocks=depth)

    def forward(self, x):
        print("Shape after entering the Decoder Module: ", x.shape)
        x = self.reversible_blocks(x)
        print("Shape after passing through reversible block in upsampling path", x.shape)
        # # Use the if-construct below if using 5 levels of resolution.
        # if self.upsample:
        #     if self.firstUpsamplingLayer:
        #         x = self.conv_first_ups(self.deconv_ups(x))
        #     else:
        #         x = self.conv_ups(self.deconv_ups(x))
        #     print("Shape after upsampling: ", x.shape)
        # Use the if-construct below if using only 4 levels of resolution.
        if self.upsample:
            x = self.conv_ups(self.deconv_ups(x))
            print("Shape after upsampling: ", x.shape)
        return x


class PartiallyReversibleUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, enc_depth=1, dec_depth=1, num_feat_maps=[]):
        super(PartiallyReversibleUnet, self).__init__()
        self.encoder_depth = enc_depth   # number of reversible blocks per sequence in the downsampling/encoder path
        self.decoder_depth = dec_depth   # number of reversible blocks per sequence in the upsampling/decoder path 
        self.levels = 4    # number of resolutions
        self.channels = num_feat_maps

        self.first_conv = spectral_norm(nn.Conv3d(in_channels, self.channels[0], kernel_size=3, padding=1, bias=False))
        self.last_conv = spectral_norm(nn.Conv3d(self.channels[0], out_channels, kernel_size=1, bias=True))
        self.tanh = nn.Tanh()

        # creating encoder levels for each resolution
        encoderModules = []
        for i in range(self.levels):
            # encoderModules.append(EncoderModule(getChannelsAtIndex(i-1), getChannelsAtIndex(i), encoder_depth, i!=0))
            encoderModules.append(EncoderModule(getChannelsAtIndex(i-1, self.channels), getChannelsAtIndex(i, self.channels), self.encoder_depth, i!=-1))
        self.encoders = nn.ModuleList(encoderModules)

        # creating decoder levels
        decoderModules = []
        for i in range(self.levels):
            # decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels-i-1), getChannelsAtIndex(self.levels-i-2),
            #                                     decoder_depth, i!=(self.levels-1)))
            
            # # use the if-else construct below if using 5 levels of resolution.
            # if i==0:
            #     decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels-i-1, self.channels), getChannelsAtIndex(self.levels-i-2, self.channels),
            #                                     self.decoder_depth, i!=(self.levels), True))
            # else:
            #     decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels-i-1, self.channels), getChannelsAtIndex(self.levels-i-2, self.channels),
            #                                     self.decoder_depth, i!=(self.levels)))
            # use the snippet below if using only 4 levels of resolution.
            decoderModules.append(DecoderModule(getChannelsAtIndex(self.levels-i-1, self.channels), getChannelsAtIndex(self.levels-i-2, self.channels),
                                                self.decoder_depth, i!=(self.levels)))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        x = self.first_conv(x)
        print("Shape after 1st Conv layer: ", x.shape)
        input_stack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            print("Shape at resolution level {} : {}\n".format(i + 1, x.shape))
            if i < self.levels-1:
                input_stack.append(x)
        print("Reached Bottleneck \n")
        for i in range(self.levels):
            x = self.decoders[i](x)
            print("Shape at upsampling level {} : {}\n".format(i + 1, x.shape))
            if i < self.levels-1:
                x = x + input_stack.pop()

        x = self.last_conv(x)
        print("Shape after Last Conv layer: ", x.shape)
        x = self.tanh(x)
        return x


class PatchGANDiscriminator(nn.Module):

    # The paper mentions: "For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a  4 × 4
    # Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a
    # convolution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky
    # ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512"

    def __init__(self, input_nc):
        super(PatchGANDiscriminator, self).__init__()

        # This PatchGAN architecture has 6 layers (1 more than the original). 46x46x46 sized patches are seen by
        # the discriminator.
        model = [spectral_norm(nn.Conv3d(input_nc, 20, kernel_size=4, stride=2, padding=1)),    # from input_nc, 64
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(20, 40, kernel_size=4, stride=2, padding=1)),    # 64, 128
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(40, 80, kernel_size=4, stride=1, padding=1)),   # 128, 256
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(80, 160, kernel_size=4, stride=1, padding=1)),   # 256, 512
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(160, 1, kernel_size=4, padding=1))]       # 512, 1

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # since a fully connected layer is not used as the output layer, an average pooling is used as a replacement
        # it is also flattened and sent as the output
        # x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
        # print(x.size(), x.size()[2:])
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0])
        return x

 
