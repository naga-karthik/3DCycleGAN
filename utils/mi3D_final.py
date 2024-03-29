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
                   'normalized_patient9_160.npy'
target_file_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/' \
                   'mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/realA2fakeB/epoch_200/0015.npy'

real_mri = np.load(source_file_path).transpose((2, 0, 1))
print(real_mri.shape)
fake_ct = np.load(target_file_path)
# print(fake_ct.shape)
if len(fake_ct.shape) != 3:
    fake_ct = fake_ct.squeeze()
print(fake_ct.shape)
# # plotting to check whether the slices match superficially at least
# plt.figure(); plt.imshow(real_mri[:, 100, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(fake_ct[:, 100, :], cmap='gray'); plt.show()
# sys.exit()


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
