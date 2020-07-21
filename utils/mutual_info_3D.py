from __future__ import division
import os, sys
import glob

from PIL import Image
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from scipy import ndimage

plt.rcParams['image.interpolation'] = 'nearest'

source_file_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/test/A/' \
                   'normalized_patient5_23.npy'
target_file_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/' \
                   'mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/realA2fakeB/epoch_200/0012.npy'

# real_images_list = sorted(glob.glob(os.path.join(source_folder_path, dataset_name, mode, domain_a) + '/*.*'))
# fake_images_list = sorted(glob.glob(os.path.join(target_folder_path, dataset_name, domain_b) + '/*.*'))
real_mri = np.load(source_file_path).transpose((2, 0, 1))
print(real_mri.shape)
fake_ct = np.load(target_file_path)
print(fake_ct.shape)
if len(fake_ct.shape) != 3:
    fake_ct = fake_ct.squeeze()
print(fake_ct.shape)

# # plotting to check whether the slices match superficially at least
# plt.figure(); plt.imshow(real_mri[:, 100, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(fake_ct[:, 100, :], cmap='gray'); plt.show()
# sys.exit()

# axial_mri, coronal_mri, sagittal_mri = [], [], []
# axial_ct, coronal_ct, sagittal_ct = [], [], []
#
# for idx in range(real_mri.shape[0]):
#     temp_mri = real_mri[idx, :, :]
#     sagittal_mri.append(temp_mri.transpose(0, 1))
#     temp_ct = fake_ct[idx, :, :]
#     sagittal_ct.append(temp_ct.transpose(0, 1))
# # plt.figure(); plt.imshow(sagittal_mri[1], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(sagittal_ct[1], cmap='gray'); plt.show()
# # # sys.exit()
#
# for idx in range(real_mri.shape[1]):
#     temp_mri = real_mri[:, idx, :]
#     axial_mri.append(temp_mri.transpose((1, 0)))
#     temp_ct = fake_ct[:, idx, :]
#     axial_ct.append(temp_ct.transpose((1, 0)))
# # plt.figure(); plt.imshow(axial_mri[20], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(axial_ct[20], cmap='gray'); plt.show()
# # # sys.exit()
#
# for idx in range(real_mri.shape[2]):
#     temp_mri = real_mri[:, :, idx]
#     coronal_mri.append(temp_mri.transpose((1, 0)))
#     temp_ct = fake_ct[:, :, idx]
#     coronal_ct.append(temp_ct.transpose((1, 0)))
# # plt.figure(); plt.imshow(coronal_mri[30], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(coronal_ct[30], cmap='gray'); plt.show()
# # sys.exit()

EPS = np.finfo(float).eps
def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array (first variable)
    y : 1D array (second variable)
    sigma: float (sigma for Gaussian smoothing of the joint histogram)
    Returns
    -------
    nmi: float (the computed similarity measure)
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi


if __name__ == '__main__':

    # mean_mutual_info_sagittal = []
    # # print(sagittal_ct[1].shape, sagittal_mri[1].shape)
    # for i in range(len(sagittal_mri)):
    #     # # Calculating and plotting 2D histogram
    #     # hist_2D, x_edges, y_edges = np.histogram2d(sagittal_mri[i].ravel(), sagittal_ct[i].ravel(), bins=20)
    #     # mi = mutual_information(histogram2d=hist_2D)
    #
    #     # Using the already available method for mutual info 2D
    #     mi = mutual_information_2d(sagittal_mri[i].ravel(), sagittal_ct[i].ravel(), normalized=True)
    #     mi = 1.0-mi
    #     # print(mi)
    #
    #     mean_mutual_info_sagittal.append(mi)
    # # print("Sum Sagittal MI score: %.4f" % (np.sum(mean_mutual_info_sagittal)))
    # print("Mean Sagittal MI score: %.4f" % (np.mean(mean_mutual_info_sagittal)))
    # # print("Mean Sagittal MI score: %.4f" % (np.mean(mean_mutual_info_sagittal)))
    #
    # mean_mutual_info_axial = []
    # for i in range(len(axial_mri)):
    #     # # Calculating and plotting 2D histogram
    #     # hist_2D, x_edges, y_edges = np.histogram2d(axial_mri[i].ravel(), axial_ct[i].ravel(), bins=20)
    #     # mi = mutual_information(histogram2d=hist_2D)
    #
    #     # Using the already available method for mutual info 2D
    #     mi = mutual_information_2d(axial_mri[i].ravel(), axial_ct[i].ravel(), normalized=True)
    #     mi = 1.0-mi
    #     # print(mi)
    #
    #     mean_mutual_info_axial.append(mi)
    # # print("Sum Axial MI score: %.4f" % (np.sum(mean_mutual_info_axial)))
    # print("Mean Axial MI score: %.4f" % (np.mean(mean_mutual_info_axial)))
    #
    # mean_mutual_info_coronal = []
    # for i in range(len(coronal_mri)):
    #     # # Calculating and plotting 2D histogram
    #     # hist_2D, x_edges, y_edges = np.histogram2d(coronal_mri[i].ravel(), coronal_ct[i].ravel(), bins=20)
    #     # mi = mutual_information(histogram2d=hist_2D)
    #
    #     # Using the already available method for mutual info 2D
    #     mi = mutual_information_2d(coronal_mri[i].ravel(), coronal_ct[i].ravel(), normalized=True)
    #     mi = 1.0-mi
    #     # print(mi)
    #     mean_mutual_info_coronal.append(mi)
    #
    # # print("Sum Coronal MI score: %.4f" % (np.sum(mean_mutual_info_coronal)))
    # print("Mean Coronal MI score: %.4f" % (np.mean(mean_mutual_info_coronal)))
    #
    # print("NMI Loss: %.4f" % (np.mean(mean_mutual_info_coronal)+
    #                           np.mean(mean_mutual_info_axial)+
    #                           np.mean(mean_mutual_info_sagittal)))

    axial_mutual_info, coronal_mutual_info, sagittal_mutual_info = [], [], []

    for idx in range(real_mri.shape[0]):
        temp_mri_sag = real_mri[idx, :, :] #.transpose(0, 1)
        temp_ct_sag = fake_ct[idx, :, :] #.transpose(0, 1)

        # Using the already available method for mutual info 2D
        # 1.0 - nmi is only used when we want to calculate loss and require gradient information
        # mi_sag = 1.0 - mutual_information_2d(temp_mri_sag.ravel(), temp_ct_sag.ravel(), normalized=True)
        # mi_sag = mutual_information_2d(temp_mri_sag.ravel(), temp_ct_sag.ravel(), normalized=True)
        # just mutual information
        mi_sag = mutual_information_2d(temp_mri_sag.ravel(), temp_ct_sag.ravel())
        sagittal_mutual_info.append(mi_sag)

    for idx in range(real_mri.shape[1]):
        temp_mri_ax = real_mri[:, idx, :] #.transpose(1, 0)
        temp_ct_ax = fake_ct[:, idx, :] #.transpose(1, 0)

        # Using the already available method for mutual info 2D
        # mi_ax = 1.0 - mutual_information_2d(temp_mri_ax.ravel(), temp_ct_ax.ravel(), normalized=True)
        # mi_ax = mutual_information_2d(temp_mri_ax.ravel(), temp_ct_ax.ravel(), normalized=True)
        # just mutual information
        mi_ax = mutual_information_2d(temp_mri_ax.ravel(), temp_ct_ax.ravel())
        axial_mutual_info.append(mi_ax)

    for idx in range(real_mri.shape[2]):
        temp_mri_cor = real_mri[:, :, idx] #.transpose(1, 0)
        temp_ct_cor = fake_ct[:, :, idx] #.transpose(1, 0)

        # Using the already available method for mutual info 2D
        # mi_cor = 1.0 - mutual_information_2d(temp_mri_cor.ravel(), temp_ct_cor.ravel(), normalized=True)
        # mi_cor = mutual_information_2d(temp_mri_cor.ravel(), temp_ct_cor.ravel(), normalized=True)
        # just mutual information
        mi_cor = mutual_information_2d(temp_mri_cor.ravel(), temp_ct_cor.ravel())
        coronal_mutual_info.append(mi_cor)

    print("Mean Sagittal MI score: %.4f" % (np.mean(sagittal_mutual_info)))
    print("Mean Axial MI score: %.4f" % (np.mean(axial_mutual_info)))
    print("Mean Coronal MI score: %.4f" % (np.mean(coronal_mutual_info)))

    # loss_nmi = np.mean(sagittal_mutual_info) + np.mean(axial_mutual_info) + np.mean(coronal_mutual_info)
    # print("NMI Loss: %.4f" % loss_nmi)
