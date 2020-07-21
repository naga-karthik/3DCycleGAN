import argparse
import sys
import os
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.insert(0, '/home/naga/PycharmProjects/deepLearning/cycleGAN3D')
# from datasets3D import ImageDataset
# from models3D_SN import ResnetGenerator3DwithSpectralNorm, UnetGenerator3DwithSpectralNorm
from datasets3D_cc import ImageDataset
# from models3D_updated_SN_cc import UnetGenerator3DUpdated
from models3D_lighter_cc import LighterUnetGenerator3D

# from cycleGAN3D.models3D_SN import UnetGenerator3DwithSpectralNorm, ResnetGenerator3DwithSpectralNorm
# from cycleGAN3D.datasets3D import ImageDataset

if __name__ == "__main__":

    # path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
    path = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/datasets/'
    # saved_path = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_models/mr2ct_new_unetUpdated_epochs250/'  # here
    saved_path = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_models/mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/'  # here
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default=path + 'mr2ct_new/', help='root directory of the dataset')

    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    # parser.add_argument('--n_cpu', type=int, default=2, help='number of CPU threads used during batch generation')

    parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
    parser.add_argument('--size_x', type=int, default=256, help='default x dimension of the image data')
    parser.add_argument('--size_y', type=int, default=128, help='default y dimension of the image data')

    parser.add_argument('--generator_A2B', type=str, default=saved_path+'netG_A2B_', help='path of A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default=saved_path+'netG_B2A_', help='path of B2A generator checkpoint file')
    args = parser.parse_args()
    print(args)

    ####### DEFINING THE VARIABLES #######
    # The Networks
    netG_A2B = LighterUnetGenerator3D(args.input_nc, args.output_nc, num_feat_maps=[20, 40, 80, 160])
    netG_B2A = LighterUnetGenerator3D(args.output_nc, args.input_nc, num_feat_maps=[20, 40 ,80, 160])
    # netG_A2B = UnetGenerator3DUpdated(args.input_nc, args.output_nc)
    # netG_B2A = UnetGenerator3DUpdated(args.output_nc, args.input_nc)
    ## using the CUDA capabilities
    netG_A2B.cuda()
    netG_B2A.cuda()

    epoch_num = [30, 80, 120, 160, 200]
    # epoch_num =  [70, 80, 90, 100]
    epch = epoch_num[4]     # here
    
    # NOTE: When a model is trained using nn.DataParallel, the models are stored in "module". Therefore, for testing, when we try to load the model, without nn.DataParallel, it throws an error (unexecpected keys in state_dict). It can be solved in two ways: (1) Temporarily load the model with nn.DataParallel (even for testing), or (2) drop the "module." that is appended to each layer like it done below.
    parallel_state_dict_A2B = torch.load(args.generator_A2B + str(epch) + '.pth', map_location='cuda')
    state_dict_A2B = OrderedDict()
    for k, v in parallel_state_dict_A2B.items():
        name = k[7:]    # removing "module."
        state_dict_A2B[name] = v

    parallel_state_dict_B2A = torch.load(args.generator_B2A + str(epch) + '.pth', map_location='cuda')
    state_dict_B2A = OrderedDict()
    for k, v in parallel_state_dict_B2A.items():
        name = k[7:]    # removing "module."
        state_dict_B2A[name] = v
    
    # Loading the saved models
    netG_A2B.load_state_dict(state_dict_A2B)
    netG_B2A.load_state_dict(state_dict_B2A)
    # netG_A2B.load_state_dict(torch.load(args.generator_A2B + str(epch) + '.pth', map_location='cuda'))
    # netG_B2A.load_state_dict(torch.load(args.generator_B2A + str(epch) + '.pth', map_location='cuda'))
    
    # netG_A2B.load_state_dict(torch.load(args.generator_A2B + '.pth'))
    # netG_B2A.load_state_dict(torch.load(args.generator_B2A + '.pth'))
    
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
    data = ImageDataset(root_dir=args.dataroot, transform=transforms_, mode='test', patches=False, one_side=True)   # here
    dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False)

    ###########################################

    ########### TESTING #############
    # creating directories to save the generated images
    # path_A = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spec_gen' \
    #          '/realA2fakeB'
    # path_B = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spec_gen' \
    #          '/realB2fakeA'
    # path_A2B = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_images/testing_generated/mr2ct_new_unetUpdated_epochs250/realA2fakeB/epoch_'+ str(epch)
    path_A2B = '/home/naga/PycharmProjects/deepLearning/cycleGAN3D/saved_images/testing_generated/mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/realA2fakeB/epoch_'+str(epch)      # here
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
        fake_B = fake_B.cpu().float().numpy()
        fake_B = fake_B.squeeze(0)

        # fake_A = netG_B2A(real_B).data

        # save these images
        # np.save(fake_A, path_B2A+'/%04d' % (i+1))
        np.save(path_A2B+'/%04d' % (i+1), fake_B)

        sys.stdout.write('\rGenerated %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')

