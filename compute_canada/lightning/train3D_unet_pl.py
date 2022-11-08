import argparse
import itertools
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/code_scripts')

from datasets3D_pl import ImageDataset
from losses_pl import gradient_consistency_loss
from utils3D_pl import LambdaLR, ReplayBuffer, weights_init_normal
from models3D_dropout_pl import LighterUnetGenerator3D, ResnetGenerator3D, PatchGANDiscriminatorwithSpectralNorm

path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/datasets/'
dataroot = path+'mr2ct_newCorrected/'

log_path = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/tensorboard_logs/'

# CHANGELOG for experiment names

# setting experiment version names
version_name = 'resnet_both'

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default=dataroot, help='root directory of the dataset')

parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch from where the learning rate starts linearly decaying to zero')

parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
# parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches')

parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--lambda_cycle', type=float, default=10.0, help='hyperparameter for tuning cycle-consistency')
parser.add_argument('--lambda_gradcon', type=float, default=5.0, help='hyperparameter for tuning gradient consistency')

parser.add_argument('--use_dropout', type=bool, default=True, help='dropout enabled by default')

parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
parser.add_argument('--size_x', type=int, default=256, help='default x dimension of the image data')
parser.add_argument('--size_y', type=int, default=128, help='default y dimension of the image data')

parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
parser.add_argument('--output_nc', type=int, default=2, help='number of output channels')
# parser.add_argument('--n_cpu', type=int, default=8, help='number of CPU threads to be used while batch generation')
args = parser.parse_args()
print(args)


class MRCTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=args.dataroot, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                 num_workers=8, shuffle=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform_ = [transforms.Normalize((0.0,), (1.0,))]
        # self.dims = (1, args.size_z, args.size_x, args.size_y)

    def prepare_data(self):
        ImageDataset(root_dir=self.data_dir, transform=self.transform_, mode='train', one_side=False)
        ImageDataset(root_dir=self.data_dir, transform=self.transform_, mode='test_1', one_side=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.mrct_train = ImageDataset(root_dir=self.data_dir, transform=self.transform_,
                                           mode='train', one_side=False)

        if stage == 'test' or stage is None:
            self.mrct_test = ImageDataset(root_dir=self.data_dir, transform=self.transform_,
                                          mode='test_1', one_side=True)

    def train_dataloader(self):
        return DataLoader(self.mrct_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mrct_test, batch_size=self.test_batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)


class LightningCycleGAN3D(pl.LightningModule):
    def __init__(self, batch_size=args.batch_size, lr=args.lr, b1=0.5, b2=0.999,
                 num_feat_maps=[32, 64, 128, 256], num_residual_blocks=9, n_epochs=args.n_epochs,
                 decay_epoch=args.decay_epoch, lambda_cycle=args.lambda_cycle, lambda_gradcon=args.lambda_gradcon,
                 use_dropout=args.use_dropout, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)

        # # instantiating the U-Net generators and the discriminators
        # self.netG_A2B = LighterUnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps,
        #                                        args.use_dropout)
        # self.netG_B2A = LighterUnetGenerator3D(args.output_nc, args.input_nc, self.hparams.num_feat_maps,
        #                                        args.use_dropout)
        # instantiating the Res-Net generators and the discriminators
        self.netG_A2B = ResnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps[0],
                                          self.hparams.num_residual_blocks, self.hparams.use_dropout)
        self.netG_B2A = ResnetGenerator3D(args.input_nc, args.outut_nc, self.hparams.num_feat_maps[0],
                                          self.hparams.num_residual_blocks, self.hparams.use_dropout)

        self.netD_A = PatchGANDiscriminatorwithSpectralNorm(args.input_nc)
        self.netD_B = PatchGANDiscriminatorwithSpectralNorm(args.input_nc)
        # initializing with proper weights
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # initialize buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.input_A = torch.Tensor(args.batch_size, args.input_nc, args.size_z, args.size_x, args.size_y).to(self.device)
        self.input_B = torch.Tensor(args.batch_size, args.output_nc, args.size_z, args.size_x, args.size_y).to(self.device)
        self.target_real = torch.Tensor(args.batch_size).fill_(1.0)  # 1 if it's real
        self.target_real.requires_grad = False
        self.target_fake = torch.Tensor(args.batch_size).fill_(0.0)  # 0 if it's fake
        self.target_fake.requires_grad = False

    def forward(self, x):
        pass

    def gan_loss(self, fake, real):
        criterion_GAN = nn.MSELoss()
        return criterion_GAN(fake, real)

    def cycle_loss(self, recovered, real):
        criterion_cycle = nn.L1Loss()
        return criterion_cycle(recovered, real)

    def aleatoric_cycle_loss(self, recovered, log_sigma, real):     # Careful what's being input! sigma or log_sigma!
        # Method 1 - Own implementation (making the model learn logSigma by itself)
        criterion_cycle = nn.L1Loss(reduction='none')  # don't take the mean now, take combined mean later
        orig_l1_loss = criterion_cycle(recovered, real)
        alea_cycle_loss = torch.abs( torch.exp(-log_sigma) * orig_l1_loss + log_sigma )
        # alea_cycle_loss = (1./torch.exp(log_sigma)) * orig_l1_loss + log_sigma
        return torch.mean(alea_cycle_loss)

    def gradcon_loss(self, real, fake):
        return gradient_consistency_loss(real_img=real, fake_img=fake)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        n_epochs = self.hparams.n_epochs
        decay_epoch = self.hparams.decay_epoch

        optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                 lr=lr, betas=(b1, b2))
        optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=lr, betas=(b1, b2))

        lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, args.epoch,
                                                                                     decay_epoch).step)
        lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, args.epoch,
                                                                                         decay_epoch).step)
        lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, args.epoch,
                                                                                         decay_epoch).step)

        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]

    def training_step(self, batch, batch_idx, optimizer_idx):
        lambda_cycle = self.hparams.lambda_cycle
        lambda_gradcon = self.hparams.lambda_gradcon

        real_A = self.input_A.copy_(batch['A']).to(self.device)
        real_B = self.input_B.copy_(batch['B']).to(self.device)

        self.target_real = self.target_real.to(self.device)
        self.target_fake = self.target_fake.to(self.device)

        # fake_A = self.netG_B2A(real_B)
        fake_A, _ = self.netG_B2A(real_B)
        fake_A.type_as(real_A)
        # fake_B = self.netG_A2B(real_A)
        fake_B, _ = self.netG_A2B(real_A)
        fake_B.type_as(real_B)

        # update generators
        if optimizer_idx == 0:
            # # showing slices on tensorboard after an epoch ends
            # # fake_B's shape: (4, 1, 48, 256, 128)
            # grid_dim1 = torchvision.utils.make_grid(fake_B[:, :, :, :, 64], nrow=4)
            # self.logger.experiment.add_image('train_synCT_slice_dim1', grid_dim1, 0)
            # grid_dim2 = torchvision.utils.make_grid(fake_B[:, :, 24, :, :], nrow=4)
            # self.logger.experiment.add_image('train_synCT_slice_dim2', grid_dim2, 0)
            # grid_dim3 = torchvision.utils.make_grid(fake_B[:, :, :, 128, :], nrow=4)
            # self.logger.experiment.add_image('train_synCT_slice_dim3', grid_dim3, 0)

            # GAN LOSS
            # fake_B = self.netG_A2B(real_A)
            pred_fake_B = self.netD_B(fake_B)
            loss_GAN_A2B = self.gan_loss(pred_fake_B, self.target_real)
            # fake_A = self.netG_B2A(real_B)
            pred_fake_A = self.netD_A(fake_A)
            loss_GAN_B2A = self.gan_loss(pred_fake_A, self.target_real)

            adversarial_loss = loss_GAN_A2B + loss_GAN_B2A
            self.log('adversarial_loss', adversarial_loss, sync_dist=True)

            # Modified CYCLE-CONSISTENCY LOSS for Aleatoric uncertainty
            recovered_A, log_sigma_a = self.netG_B2A(fake_B)
            loss_cycle_A2B2A = self.aleatoric_cycle_loss(recovered_A, log_sigma_a, real_A)
            recovered_B, log_sigma_b = self.netG_A2B(fake_A)
            loss_cycle_B2A2B = self.aleatoric_cycle_loss(recovered_B, log_sigma_b, real_B)

            # # ORIGINAL CYCLE-CONSISTENCY LOSS
            # recovered_A = self.netG_B2A(fake_B)
            # loss_cycle_A2B2A = self.cycle_loss(recovered_A, real_A)
            # recovered_B = self.netG_A2B(fake_A)
            # loss_cycle_B2A2B = self.cycle_loss(recovered_B, real_B)

            loss_cycle_consistency = lambda_cycle * (loss_cycle_A2B2A + loss_cycle_B2A2B)
            self.log('cycle_loss', loss_cycle_consistency, sync_dist=True)

            # GRADIENT-CONSISTENCY LOSS
            loss_gc_A = 0.5 * self.gradcon_loss(real=real_A, fake=fake_B)
            loss_gc_B = 0.5 * self.gradcon_loss(real=real_B, fake=fake_A)

            loss_gradient_consistency = lambda_gradcon * (loss_gc_A + loss_gc_B)
            self.log('gradient_consistency_loss', loss_gradient_consistency)

            # COMBINING ALL LOSSES FOR THE GENERATOR
            loss_G = adversarial_loss + loss_cycle_consistency + loss_gradient_consistency
            self.log("generators_loss", loss_G, prog_bar=True, sync_dist=True)
            return loss_G

        # update discriminator D_A
        if optimizer_idx == 1:
            # Real Loss
            pred_real_A = self.netD_A(real_A)
            loss_D_real_A = self.gan_loss(pred_real_A, self.target_real)
            # Fake Loss
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = self.netD_A(fake_A.detach())
            loss_D_fake_A = self.gan_loss(pred_fake_A, self.target_fake)
            # Total Loss
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            self.log('discriminator_A_loss', loss_D_A, prog_bar=True, sync_dist=True)
            return loss_D_A

        # update discriminator D_B
        if optimizer_idx == 2:
            # Real Loss
            pred_real_B = self.netD_B(real_B)
            loss_D_real_B = self.gan_loss(pred_real_B, self.target_real)
            # Fake Loss
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = self.netD_B(fake_B.detach())
            loss_D_fake_B = self.gan_loss(pred_fake_B, self.target_fake)
            # Total Loss
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            self.log("discriminator_B_loss", loss_D_B, prog_bar=True, sync_dist=True)
            return loss_D_B

    def test_step(self, batch, batch_idx):
        real_A = self.input_A.copy_(batch['A']).to(self.device)
        fake_B = self.netG_A2B(real_A)

        # showing slices on tensorboard during testing
        # fake_B's shape: (4, 1, 48, 256, 128)
        grid_dim1 = torchvision.utils.make_grid(fake_B[:, :, :, :, 64], nrow=4)
        self.logger.experiment.add_image('test_synCT_slice_dim1', grid_dim1, batch_idx)
        grid_dim2 = torchvision.utils.make_grid(fake_B[:, :, 24, :, :], nrow=4)
        self.logger.experiment.add_image('test_synCT_slice_dim2', grid_dim2, batch_idx)
        grid_dim3 = torchvision.utils.make_grid(fake_B[:, :, :, 128, :], nrow=4)
        self.logger.experiment.add_image('test_synCT_slice_dim3', grid_dim3, batch_idx)


if __name__ == '__main__':

    n_gpus = torch.cuda.device_count()
    dm = MRCTDataModule()
    # snippet for training
    model = LightningCycleGAN3D()
    tb_logger = TensorBoardLogger(log_path, name='cyclegan3d_16bit', version_name=version_name)
    trainer = pl.Trainer(precision=16, gpus=n_gpus, distributed_backend='ddp', max_epochs=3, logger=tb_logger,
                         progress_bar_refresh_rate=30)
    trainer.fit(model, dm)

    # # only testing
    # saved_path = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/tensorboard_logs/' \
    #              'cyclegan3d_16bit/version_5/checkpoints/epoch=0.ckpt'
    # params_path = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/tensorboard_logs/' \
    #               'cyclegan3d_16bit/version_5/hparams.yaml'
    # loc = 'cpu'
    # model = LightningCycleGAN3D.load_from_checkpoint(saved_path, map_location=loc)
    # trainer_for_testing = pl.Trainer(precision=32)
    # print("Only Testing!")
    # trainer_for_testing.test(model, dm)

    # # Continuing training from a checkpoint
    # need to use the argument 'resume_from_checkpoint' in Trainer
    # PATH = '/home/karthik7/projects/def-laporte1/karthik7/lightning/cycleGAN_lightning/tensorboard_logs/cyclegan_a2o/version_5/checkpoints/epoch=4.ckpt'
    # hparams_file = '/home/karthik7/projects/def-laporte1/karthik7/lightning/cycleGAN_lightning/tensorboard_logs/cyclegan_a2o/version_5/hparams.yaml'
    # model = LightningCycleGAN3D.load_from_checkpoint(PATH, hparams_file)
    # trainer = pl.Trainer(precision=16, gpus=2, distributed_backend='ddp', max_epochs=200, progress_bar_refresh_rate=20,
    #                      logger=tb_logger, log_gpu_memory='all')
    # trainer.fit(model, dm)



