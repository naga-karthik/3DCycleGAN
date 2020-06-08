import argparse
import sys
import os
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.insert(0, '/home/naga/PycharmProjects/deepLearning/cycleGAN3D')
from datasets3D import ImageDataset
from models3D_SN import ResnetGenerator3DwithSpectralNorm, UnetGenerator3DwithSpectralNorm

# from cycleGAN3D.models3D_SN import UnetGenerator3DwithSpectralNorm, ResnetGenerator3DwithSpectralNorm
# from cycleGAN3D.datasets3D import ImageDataset

if __name__ == "__main__":

    # path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
    path = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/datasets/'
    saved_path = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_models/mr2ct_spectral/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default=path + 'mr2ct/', help='root directory of the dataset')

    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of CPU threads used during batch generation')

    parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
    parser.add_argument('--size_x', type=int, default=256, help='default x dimension of the image data')
    parser.add_argument('--size_y', type=int, default=128, help='default y dimension of the image data')

    parser.add_argument('--generator_A2B', type=str, default=saved_path+'netG_A2B.pth', help='path of A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default=saved_path+'netG_B2A.pth', help='path of B2A generator checkpoint file')
    args = parser.parse_args()
    print(args)

    ####### DEFINING THE VARIABLES #######
    # The Networks
    netG_A2B = UnetGenerator3DwithSpectralNorm(args.input_nc, args.output_nc)
    netG_B2A = UnetGenerator3DwithSpectralNorm(args.output_nc, args.input_nc)

    ## using the CUDA capabilities
    netG_A2B.cuda()
    netG_B2A.cuda()

    # Loading the saved models
    netG_A2B.load_state_dict(torch.load(args.generator_A2B))
    netG_B2A.load_state_dict(torch.load(args.generator_B2A))

    # Setting the model to 'test' mode
    netG_A2B.eval()
    netG_B2A.eval()

    # SETTING INPUTS AND ALLOCATION MEMORY
    Tensor = torch.cuda.FloatTensor
    # Tensor = torch.FloatTensor
    input_A = Tensor(args.batch_size, args.input_nc, args.size_z, args.size_x, args.size_y)
    input_B = Tensor(args.batch_size, args.output_nc, args.size_z, args.size_x, args.size_y)

    # Loading the data
    transforms_ = [transforms.Normalize((0.0,), (1.0,))]
    data = ImageDataset(root_dir=args.dataroot, transform=transforms_, mode='test', patches=False, one_side=True)
    dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False)

    ###########################################

    ########### TESTING #############
    # creating directories to save the generated images
    # path_A = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spec_gen' \
    #          '/realA2fakeB'
    # path_B = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spec_gen' \
    #          '/realB2fakeA'
    path_A2B = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_images/testing_generated/mr2ct_spectral' \
             '/realA2fakeB'
    if not os.path.exists(path_A2B):
        os.makedirs(path_A2B)
    # path_B2A = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_images/testing_generated/mr2ct_spec_gen' \
    #          '/realB2fakeA'
    # if not os.path.exists(path_B2A):
    #     os.makedirs(path_B2A)

    for i, batch in enumerate(dataloader):
        # set model input
        real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))

        # generate images
        fake_B = netG_A2B(real_A).data
        # fake_A = netG_B2A(real_B).data

        # save these images
        # np.save(fake_A, path_B2A+'/%04d' % (i+1))
        np.save(fake_B, path_A2B+'/%04d' % (i+1))

        sys.stdout.write('\rGenerated %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')

