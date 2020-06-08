# Test file for testing out new models by sending a small random input and checking their output.
import sys
sys.path.insert(0, '/home/naga/PycharmProjects/deepLearning/cycleGAN3D')

import torch
# from models3D import UnetGenerator3D, ResnetGenerator3D, PatchGANDiscriminator
from models3D_SN import UnetGenerator3DwithSpectralNorm, ResnetGenerator3DwithSpectralNorm, \
    PatchGANDiscriminatorwithSpectralNorm

# from cycleGAN3D.models3D import UnetGenerator3D, ResnetGenerator3D, PatchGANDiscriminator
# from cycleGAN3D.models3D_SN import UnetGenerator3DwithSpectralNorm, ResnetGenerator3DwithSpectralNorm, \
#     PatchGANDiscriminatorwithSpectralNorm

# from cycleGAN.models_updated import UnetGenerator
# from cycleGAN.reimplementations.unet import Unet


def Unetmodel():
    # return UnetGenerator3D(in_channels=1, out_channels=1)
    return UnetGenerator3DwithSpectralNorm(in_channels=1, out_channels=1)
    # return Unet(in_channels=1, out_channels=1, init_featMaps=32)

def Resnetmodel():
    # return ResnetGenerator3D(input_nc=1, output_nc=1, num_residual_blocks=2)
    return ResnetGenerator3DwithSpectralNorm(input_nc=1, output_nc=1, num_residual_blocks=2)

def PatchGAN():
    return PatchGANDiscriminatorwithSpectralNorm(input_nc=1)

if __name__ == "__main__":
    # net = Unetmodel()
    net = Resnetmodel()
    # net = PatchGAN()
    # input = torch.randn(2, 1, 48, 256, 128)   # for 3D Resnet, laptop not able to handle this much memory -> hung
    input = torch.randn(2, 1, 48, 256, 128)
    # input = torch.randn(8, 1, 6, 6, 6)
    y = net(input)
    # y = net(torch.randn(1, 1, 256, 256))
    print(y.shape)
