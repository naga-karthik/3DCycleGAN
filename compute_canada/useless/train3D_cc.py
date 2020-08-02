import argparse
import itertools
import os, sys
import numpy as np
# sys.path.insert(0, '/home/naga/PycharmProjects/deepLearning/cycleGAN3D')
sys.path.insert(0, '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/code_scripts')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from datasets3D_cc import ImageDataset
from utils3D_cc import Logger, LambdaLR, ReplayBuffer, weights_init_normal #, tensor2image
from models3D_SN_cc import ResnetGenerator3DwithSpectralNorm, UnetGenerator3DwithSpectralNorm, \
    PatchGANDiscriminatorwithSpectralNorm

path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/datasets/'

save_path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_models/mr2ct_resnet'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# paths for saving the images generated during training
traingen_path_A2B = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_images/training_generated' \
                    '/mr2ct_resnet/realA2fakeB'
traingen_path_B2A = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/saved_images/training_generated' \
                    '/mr2ct_resnet/realB2fakeA'
if not os.path.exists(traingen_path_A2B):
    os.makedirs(traingen_path_A2B)
if not os.path.exists(traingen_path_B2A):
    os.makedirs(traingen_path_B2A)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default=path + 'mr2ct/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from where the learning rate starts linearly decaying to zero')
parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
parser.add_argument('--size_x', type=int, default=256, help='default x dimension of the image data')
parser.add_argument('--size_y', type=int, default=128, help='default y dimension of the image data')
parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
parser.add_argument('--num_residual_blocks', type=int, default=9, help='number of residual blocks')
# parser.add_argument('--continue_train', action='store_true', help='continue training - load the last saved model')
args = parser.parse_args()
print(args)

###### DEFINING VARIABLES FOR THE GENERATOR AND THE DISCRIMINATOR ######
# THE NETWORKS
# netG_A2B = UnetGenerator3D(args.input_nc, args.output_nc)
# netG_B2A = UnetGenerator3D(args.output_nc, args.input_nc)
netG_A2B = ResnetGenerator3DwithSpectralNorm(args.input_nc, args.output_nc, args.num_residual_blocks)
netG_B2A = ResnetGenerator3DwithSpectralNorm(args.output_nc, args.input_nc, args.num_residual_blocks)
netD_A = PatchGANDiscriminatorwithSpectralNorm(args.input_nc)
netD_B = PatchGANDiscriminatorwithSpectralNorm(args.output_nc)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     netG_A2B = nn.DistributedDataParallel(netG_A2B)
#     netG_B2A = nn.DataParallel(netG_B2A)
#     netD_A = nn.DataParallel(netD_A)
#     netD_B = nn.DataParallel(netD_B)

# using cuda
netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()

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
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

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
input_A = Tensor(args.batch_size, args.input_nc, args.size_z, args.size_x, args.size_y) # defining the size of the input here
input_B = Tensor(args.batch_size, args.output_nc, args.size_z, args.size_x, args.size_y) # defining the size of the output here
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False) # 1 if it's real
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False) # 0 if it's fake

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# LOADING THE DATASET
transforms_ = [transforms.Normalize((0.0,), (1.0,))]
data = ImageDataset(root_dir=args.dataroot, transform=transforms_, patches=False)
dataloader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True)

# PLOTTING THE LOSS
logger = Logger(args.epoch, args.n_epochs, len(dataloader))

####### TRAINING ##### (THE MOST IMPORTANT PART)

j=0
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

        # CYCLE-CONSISTENCY LOSS
        recovered_A = netG_B2A(fake_B)
        loss_cycle_A2B2A = criterion_cycle(recovered_A, real_A)*10.0
        recovered_B = netG_A2B(fake_A)
        loss_cycle_B2A2B = criterion_cycle(recovered_B, real_B)*10.0

        # IDENTITY LOSS Identity loss in the paper is defined like this: lambda_identity * (||G_A(B) - B|| * lambda_B
        # + ||G_B(A) - A|| * lambda_A) -> Here: ||netG_A2B(B) - B||*(10*0.5) + ||netG_B2A(A) - A||*(10*0.5)
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # Idea: When real A is fed to netG_B2A, the output should be A! (i.e. netG_B2A(A) = A)
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # COMBINING ALL LOSSES FOR THE GENERATOR NOW
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B2A + loss_cycle_B2A2B + loss_identity_A + loss_identity_B
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
        ## "we divide the objective by 2 while optimizing D, which slows down the rate at which D learns,
        ## relative to the rate of G"
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

        # saving the generated images for every 100th image in a batch
        if (i+1) % 700 == 0:
            j+=1
            tempB = fake_B[0].cpu().double().numpy()
            tempB.squeeze(0)
            np.save(traingen_path_A2B +'/%04d' % j, tempB)

            tempA = fake_A[0].cpu().double().numpy()
            tempA.squeeze(0)
            np.save(traingen_path_B2A + '/%04d' % j, tempA)
            # tempA = tensor2image(fake_A)
            # imageio.imsave(traingen_path_B2A +'/%04d.png' % j, tempA.transpose(1,2,0))

        # Show progess on Terminal
        logger.log(save_path, {'G Loss': loss_G,
                               'G Adversarial Loss': (loss_GAN_A2B + loss_GAN_B2A),
                               'G Cycle-Consistency Loss': (loss_cycle_A2B2A + loss_cycle_B2A2B),
                               'G Identity Loss': (loss_identity_A + loss_identity_B),
                               'D Loss': (loss_D_A + loss_D_B)})

    # UPDATE LEARNING RATES
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # SAVE MODELS (at the end of every 10th epoch)
    if (epoch+1) % 1 == 0:
        torch.save(netG_A2B.state_dict(), save_path+"/netG_A2B_%d.pth" % (epoch+1))
        torch.save(netG_B2A.state_dict(), save_path+"/netG_B2A_%d.pth" % (epoch+1))
        torch.save(netD_A.state_dict(), save_path+"/netD_A_%d.pth" % (epoch+1))
        torch.save(netD_B.state_dict(), save_path+"/netD_B_%d.pth" % (epoch+1))