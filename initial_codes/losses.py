import torch
import torch.nn as nn
from torchvision import models
import scipy.ndimage as ndimage
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.filters import sobel_h, sobel_v


def gradient_consistency_loss(real_img, fake_img):
    gradx_img_a = torch.DoubleTensor(ndimage.sobel(real_img, axis=0))  # axis=0 is the x-axis
    gradx_img_b = torch.DoubleTensor(ndimage.sobel(fake_img, axis=0))
    ncc_x = normalized_cross_correlation(gradx_img_a, gradx_img_b)

    grady_img_a = torch.DoubleTensor(ndimage.sobel(real_img, axis=1))  # axis=1 is the y-axis
    grady_img_b = torch.DoubleTensor(ndimage.sobel(fake_img, axis=1))
    ncc_y = normalized_cross_correlation(grady_img_a, grady_img_b)

    gradz_img_a = torch.DoubleTensor(ndimage.sobel(real_img, axis=2))  # axis=2 is the z-axis
    gradz_img_b = torch.DoubleTensor(ndimage.sobel(fake_img, axis=2))
    ncc_z = normalized_cross_correlation(gradz_img_a, gradz_img_b)

    grad_corr_ab = 0.5 * (ncc_x + ncc_y + ncc_z)
    result = (1.0 - grad_corr_ab)

    # -- For testing with a few images, whether ndimage.sobel and skimage.filters.sobel are the same.---
    # tempx = ndimage.sobel(real_img, axis=0)
    # plt.figure(); plt.imshow(real_img[20, :, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(tempx[20, :, :], cmap='gray'); plt.show()

    # tempy = ndimage.sobel(real_img, axis=1)
    # tempy1 = ndimage.sobel(fake_img, axis=1)
    # plt.figure(); plt.imshow(tempy[20, :, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(tempy1[20, :, :], cmap='gray'); plt.show()

    # tempz = ndimage.sobel(real_img, axis=2)
    # plt.figure(); plt.imshow(tempz[20, :, :], cmap='gray'); plt.show()

    return torch.mean(result)


def normalized_cross_correlation(img_a, img_b):
    mu_a = torch.mean(img_a)
    mu_b = torch.mean(img_b)
    sigma_a = torch.std(img_a)
    sigma_b = torch.std(img_b)

    numerator = torch.mean((img_a - mu_a) * (img_b - mu_b))
    denominator = sigma_a * sigma_b
    ncc_ab = numerator / (denominator + 1e-6)  # to avoid division by zero, in case

    return ncc_ab


class FeatureLossNet:
    def __init__(self, device):
        self.vgg16 = models.vgg16(pretrained=True).features.to(device).eval()
        self.pool2_features = []
        self.pool5_features = []
        self.vgg16[9].register_forward_hook(self.hook)
        self.transform = Transform(device).to(device)
        self.criterion_perceptual = nn.L1Loss()

    def hook(self, model, input, output):
        self.pool2_features.append(output.data)

    def perceptual_loss(self, real_img, cyc_rec_img):
        """Note the the real and cycle image have to be in the range [-1, 1]
        And two more things: VGG takes 2D images only and that too with RGB channels"""

        # # bringing the image from [0,1] to [-1,1]. Remember that all the data was normalized to lie between [0,1]
        # # by using transforms.Normalize
        # real_img = (real_img + 1.0) / 2.0
        # cyc_rec_img = (cyc_rec_img + 1.0) / 2.0

        # current image dimension: [batch_size, channels, z, x, y] -> [batch_size, 1, 48, 256, 128]
        # required image dimension for vgg: [batch_size, 3, 48, 256, 128]
        real_img = real_img.repeat(1, 3, 1, 1, 1)
        cyc_rec_img = cyc_rec_img.repeat(1, 3, 1, 1, 1)

        # since vgg works with 2D images, we need to convert the 3D image into individual slices and calculate
        # the perceptual loss individually for each pair of slices
        # print(real_img.shape)
        len_z, len_x, len_y = real_img.shape[2], real_img.shape[3], real_img.shape[4]
        res_z, res_x, res_y = 0.0, 0.0, 0.0
        combined_result = 0.0

        for i in range(len_z):
            self.pool2_features.clear()
            self.pool5_features.clear()

            trsfmd_real_img = self.transform(torch.squeeze(real_img[:, :, i, :, :], 2))
            # print(trsfmd_real_img.min(), trsfmd_real_img.max())
            # print(trsfmd_real_img.shape)
            real_img_feats = self.vgg16(trsfmd_real_img)
            self.pool5_features.append(real_img_feats)
            trsfmd_cycle_img = self.transform(torch.squeeze(cyc_rec_img[:, :, i, :, :], 2))
            cycle_img_feats = self.vgg16(trsfmd_cycle_img)
            self.pool5_features.append(cycle_img_feats)

            l1 = self.criterion_perceptual(self.pool2_features[1], self.pool2_features[0].detach())
            l2 = self.criterion_perceptual(self.pool5_features[1], self.pool5_features[1].detach())
            res_z += (l1 + l2)/len_z

        for i in range(len_x):
            self.pool2_features.clear()
            self.pool5_features.clear()

            trsfmd_real_img = self.transform(torch.squeeze(real_img[:, :, :, i, :], 3))
            real_img_feats = self.vgg16(trsfmd_real_img)
            self.pool5_features.append(real_img_feats)
            trsfmd_cycle_img = self.transform(torch.squeeze(cyc_rec_img[:, :, :, i, :], 3))
            cycle_img_feats = self.vgg16(trsfmd_cycle_img)
            self.pool5_features.append(cycle_img_feats)

            l1 = self.criterion_perceptual(self.pool2_features[1], self.pool2_features[0].detach())
            l2 = self.criterion_perceptual(self.pool5_features[1], self.pool5_features[1].detach())
            res_x += (l1 + l2)/len_x

        for i in range(len_y):
            self.pool2_features.clear()
            self.pool5_features.clear()

            trsfmd_real_img = self.transform(torch.squeeze(real_img[:, :, :, :, i], 4))
            real_img_feats = self.vgg16(trsfmd_real_img)
            self.pool5_features.append(real_img_feats)
            trsfmd_cycle_img = self.transform(torch.squeeze(cyc_rec_img[:, :, :, :, i], 4))
            cycle_img_feats = self.vgg16(trsfmd_cycle_img)
            self.pool5_features.append(cycle_img_feats)

            l1 = self.criterion_perceptual(self.pool2_features[1], self.pool2_features[0].detach())
            l2 = self.criterion_perceptual(self.pool5_features[1], self.pool5_features[1].detach())
            res_y += (l1 + l2)/len_y

        combined_result = res_z + res_x + res_y
        return torch.sum(combined_result)


class Transform(nn.Module):
    def __init__(self, device, cnn_normalization_mean=torch.Tensor([0.485, 0.456, 0.406]),
                 cnn_normalization_std=torch.Tensor([0.229, 0.224, 0.225])):
        super(Transform, self).__init__()
        self.mean = torch.Tensor(cnn_normalization_mean).to(device).view(-1, 1, 1)
        self.std = torch.Tensor(cnn_normalization_std).to(device).view(-1, 1, 1)

    def forward(self, img):
        img = (img + 1) / 2
        return (img - self.mean) / self.std


if __name__ == "__main__":
    real_img_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/sample_mr2ct/train/A' \
                    '/normalized_01_wat_crpd.npy'
    fake_img_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/sample_mr2ct/train/B' \
                    '/normalized_anon_crpd.npy'
    img1 = torch.FloatTensor(np.load(real_img_path)).permute((2,0,1)).unsqueeze(0).unsqueeze(0)
    img2 = torch.FloatTensor(np.load(fake_img_path)).permute((2,0,1)).unsqueeze(0).unsqueeze(0)

    # gc = gradient_consistency_loss(img1, img2)
    # print(gc)

    device = 'cpu'
    netFeatureLoss = FeatureLossNet(device)
    img1 = (img1 + 1.0)/2.0
    img2 = (img2 + 1.0)/2.0

    perceptual = netFeatureLoss.perceptual_loss(img1, img2)

    # img = np.load(real_img_path)
    # img = img[:, :, 20]
    # gradx_img = sobel_h(img)
    # grady_img = sobel_v(img)
    # plt.figure(); plt.imshow(img, cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(gradx_img, cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(grady_img, cmap='gray'); plt.show()
