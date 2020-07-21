import argparse
import itertools
import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

# packages for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# sys.path.insert(0, '/home/naga/PycharmProjects/deepLearning/cycleGAN3D')
sys.path.insert(0, '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/code_scripts')

from datasets3D_cc import ImageDataset
from losses_cc import FeatureLossNet
from utils3D_cc import Logger, LambdaLR, ReplayBuffer, weights_init_normal
from models3D_updated_SN_cc import UnetGenerator3DUpdated, PatchGANDiscriminatorwithSpectralNorm
from models3D_SN_cc import UnetGenerator3DwithSpectralNorm

path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/datasets/'

save_path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_models/mr2ct_unet_perceptual'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# paths for saving the images generated during training
traingen_path_A2B = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_images/training_generated' \
                    '/mr2ct_unet_perceptual/realA2fakeB'
traingen_path_B2A = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_images/training_generated' \
                    '/mr2ct_unet_perceptual/realB2fakeA'
if not os.path.exists(traingen_path_A2B):
    os.makedirs(traingen_path_A2B)
if not os.path.exists(traingen_path_B2A):
    os.makedirs(traingen_path_B2A)


# ------ DEFINING VARIABLES FOR THE GENERATOR AND THE DISCRIMINATOR -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_model(rank, args):

    print(f"Running Distributed 3D CycleGAN on rank {rank}.")
    setup(rank, args.world_size)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # THE NETWORKS
    # netG_A2B = UnetGenerator3DUpdated(args.input_nc, args.output_nc)
    # netG_B2A = UnetGenerator3DUpdated(args.output_nc, args.input_nc)
    netG_A2B = UnetGenerator3DwithSpectralNorm(args.input_nc, args.output_nc).to(rank)
    netG_B2A = UnetGenerator3DwithSpectralNorm(args.output_nc, args.input_nc).to(rank)
    netD_A = PatchGANDiscriminatorwithSpectralNorm(args.input_nc).to(rank)
    netD_B = PatchGANDiscriminatorwithSpectralNorm(args.output_nc).to(rank)
    # defining the network for calculating feature loss
    netFeatureLoss = FeatureLossNet(device='cuda')

    # wraps the network around distributed package
    netG_A2B = DDP(netG_A2B, device_ids=[rank])
    netG_B2A = DDP(netG_B2A, device_ids=[rank])
    netD_B = DDP(netD_B, device_ids=[rank])
    netD_A = DDP(netD_A, device_ids=[rank])

    if args.epoch != 1:  # in this case, go to the folder to see at which epoch training stopped and run the model again
                         # by specifying --epoch='the-last-epoch'
        # load saved models
        netG_A2B.load_state_dict(torch.load(save_path+"/netG_A2B_%d.pth" % (args.epoch)))
        netG_B2A.load_state_dict(torch.load(save_path+"/netG_B2A_%d.pth" % (args.epoch)))
        netD_A.load_state_dict(torch.load(save_path+"/netD_A_%d.pth" % (args.epoch)))
        netD_B.load_state_dict(torch.load(save_path+"/netD_B_%d.pth" % (args.epoch)))
    else:
        # initializing with proper weights
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

    # THE LOSSES
    criterion_GAN = nn.MSELoss().to(rank)
    criterion_cycle = nn.L1Loss().to(rank)
    criterion_identity = nn.L1Loss().to(rank)   # without this, the generator tends to change the "tint" of the output images.
    # criterion_gradient_consistency =

    # THE OPTIMIZERS AND LEARNING RATE SCHEDULERS
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    # INPUTS AND MEMORY ALLOCATION
    Tensor = torch.cuda.FloatTensor   # this is float32
    # Tensor = torch.cuda.DoubleTensor    # this is float64
    input_A = Tensor(args.batch_size, args.input_nc, args.size_z, args.size_x, args.size_y)     # defining the input size
    input_B = Tensor(args.batch_size, args.output_nc, args.size_z, args.size_x, args.size_y)    # defining the input size
    target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)     # 1 if it's real
    target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)     # 0 if it's fake

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [transforms.Normalize((0.0,), (1.0,))]
    data = ImageDataset(root_dir=args.dataroot, transform=transforms_, patches=True)
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=data, num_replicas=args.world_size, rank=rank)
    dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, pin_memory=True, sampler=data_sampler)

    # PLOTTING THE LOSS
    # logger = Logger(args.n_epochs, len(dataloader))
    logger = Logger(args.epoch, args.n_epochs, len(dataloader))

    # ------ TRAINING ------# (THE MOST IMPORTANT PART)
    j = 0
    for epoch in range(args.epoch, args.n_epochs+1):

        for i, batch in enumerate(dataloader):
            # setting the model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ####### the two generators: A2B and B2A #######
            optimizer_G.zero_grad()

            # GAN LOSS
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)
            adversarial_loss = loss_GAN_A2B + loss_GAN_B2A

            # CYCLE-CONSISTENCY LOSS
            recovered_A = netG_B2A(fake_B)
            loss_cycle_A2B2A = criterion_cycle(recovered_A, real_A)   # lambda_cycle = 3.0
            recovered_B = netG_A2B(fake_A)
            loss_cycle_B2A2B = criterion_cycle(recovered_B, real_B)
            cycle_consistency_loss = 10.0*(loss_cycle_A2B2A + loss_cycle_B2A2B)

            # # IDENTITY LOSS
            # same_B = netG_A2B(real_B)
            # loss_identity_B = criterion_identity(same_B, real_B)
            # same_A = netG_B2A(real_A)
            # loss_identity_A = criterion_identity(same_A, real_A)
            # identity_loss = 5.0*(loss_identity_A + loss_identity_B)

            # PERCEPTUAL LOSS
            # Penalizing the deviation from high-dimensional feature space for real and cycle-reconstructed images.
            loss_perceptual_A = netFeatureLoss.perceptual_loss(real_A, recovered_A)
            loss_perceptual_B = netFeatureLoss.perceptual_loss(real_B, recovered_B)
            perceptual_loss = 0.5 * (loss_perceptual_A + loss_perceptual_B)     # 0.5 = 0.05*lambda-cycle (=10.0)

            # COMBINING ALL LOSSES FOR THE GENERATOR NOW
            # loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B2A + loss_cycle_B2A2B + loss_identity_A + loss_identity_B
            loss_G = adversarial_loss + cycle_consistency_loss + perceptual_loss
            loss_G.backward()

            optimizer_G.step()
            #####################################

            ###### DISCRIMINATOR A ######
            optimizer_D_A.zero_grad()

            # Real Loss
            pred_real_A = netD_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, target_real)

            # Fake Loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = netD_A(fake_A.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)

            # Total Loss
            loss_D_A = (loss_D_real_A + loss_D_fake_A)*0.5
            # Why mulitply by 0.5? It is mentioned in the paper as:
            # "we divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
            # relative to the rate of G"
            loss_D_A.backward()

            optimizer_D_A.step()
            ################################

            ####### DISCRIMINATOR B #######
            optimizer_D_B.zero_grad()

            # Real Loss
            pred_real_B = netD_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, target_real)

            # Fake Loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = netD_B(fake_B.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)

            # Total Loss
            loss_D_B = (loss_D_real_B + loss_D_fake_B)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ################################

            # Show progess on Terminal
            logger.log(save_path, {'G Loss': loss_G,
                                   'G Adversarial Loss': adversarial_loss,
                                   'G Cycle-Consistency Loss': cycle_consistency_loss,
                                   # 'G Identity Loss': identity_loss,
                                   'G Perceptual Loss': perceptual_loss,
                                   'D Loss': (loss_D_A + loss_D_B)})

            # saving the generated images for every 100th image in a batch
            if (i+1) % 300 == 0:
                j += 1
                tempB = fake_B[0]
                # print(tempB.shape, tempB.dtype, fake_B[0].shape)

                # Code for reshaping the patches back to the original image
                orig_B = tempB.view(1, 4, 2, 48, 64, 64)
                outd = orig_B.shape[0] * orig_B.shape[3]
                outh = orig_B.shape[1] * orig_B.shape[4]
                outw = orig_B.shape[2] * orig_B.shape[5]
                orig_B = orig_B.unsqueeze(0).permute(0, 1,4, 2,5, 3,6).contiguous()
                # # print(original_A.shape)
                orig_B = orig_B.view(1, outd, outh, outw)
                # # print((original_A == item_A[:, :outd, :outh, :outw]).all())
                tempB = orig_B.cpu().double().numpy()
                print(tempB.shape)
                np.save(traingen_path_A2B +'/%04d' % j, tempB)


                tempA = fake_A[0]
                orig_A = tempA.view(1, 4, 2, 48, 64, 64)
                outd = orig_A.shape[0] * orig_A.shape[3]
                outh = orig_A.shape[1] * orig_A.shape[4]
                outw = orig_A.shape[2] * orig_A.shape[5]
                orig_A = orig_A.unsqueeze(0).permute(0, 1,4, 2,5, 3,6).contiguous()
                # # print(original_A.shape)
                orig_A = orig_A.view(1, outd, outh, outw)
                # # print((original_A == item_A[:, :outd, :outh, :outw]).all())
                tempA = orig_A.cpu().double().numpy()
                np.save(traingen_path_B2A + '/%04d' % j, tempA)

                # tempA = tensor2image(fake_A)
                # imageio.imsave(traingen_path_B2A +'/%04d.png' % j, tempA.transpose(1,2,0))

        # UPDATE LEARNING RATES
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # SAVE MODELS (at the end of every epoch)
        if (epoch+1) % 10 == 0:
            torch.save(netG_A2B.state_dict(), save_path+"/netG_A2B_%d.pth" % (epoch+1))
            torch.save(netG_B2A.state_dict(), save_path+"/netG_B2A_%d.pth" % (epoch+1))
            torch.save(netD_A.state_dict(), save_path+"/netD_A_%d.pth" % (epoch+1))
            torch.save(netD_B.state_dict(), save_path+"/netD_B_%d.pth" % (epoch+1))

    cleanup()


def run_train_model(train_func, world_size):

    parser = argparse.ArgumentParser()
    # parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
    # parser.add_argument('--gpus', type=int, default=2, help='number of GPUs per node')
    # parser.add_argument('--nrank', type=int, default=0, help='ranking within the nodes')

    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs for training')

    parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')

    parser.add_argument('--dataroot', type=str, default=path + 'mr2ct_new/', help='root directory of the dataset')

    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=75,
                        help='epoch from where the learning rate starts linearly decaying to zero')
    parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
    parser.add_argument('--size_x', type=int, default=64, help='default x dimension of the image data')
    parser.add_argument('--size_y', type=int, default=64, help='default y dimension of the image data')

    # The patches are created in such a way that there are 8 patches of size 48x64x64 for 1 input volume of size
    # 48x256x128. In other words, 48x256x128 = 8x48x64x64. Now, the problem is that nn.Conv3d requires inputs in the
    # following format: (N,C,D,H,W). Therefore, by changing the input channels and output channels below to 8, we can
    # bring our input to this format (i.e. N(batch size), 8, 48, 64, 64). Instead of starting out with 1 channel and
    # processing the entire volume as a whole (i.e N, 1, 48, 256, 128), patches ease the load on the memory by dividing
    # the input in this format: (N, 8, 48, 256, 128). The output size will be the same as shown previously and they
    # can be reconstructed to form the original volume.
    parser.add_argument('--input_nc', type=int, default=8, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=8, help='number of output channels')

    parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to be used while batch generation')
    # parser.add_argument('--continue_train', action='store_true', help='continue training - load the last saved model')
    args = parser.parse_args()
    print(args)

    mp.spawn(train_func, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    run_train_model(train_model, n_gpus)
