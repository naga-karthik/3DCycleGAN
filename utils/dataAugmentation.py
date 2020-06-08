import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from elasticdeform import deform_random_grid
import nrrd
import matplotlib.pyplot as plt
import os, sys, glob, random

# Summary of what this code is doing:
# Loads the data from respective folders and augments the data using 10 random rotations for each volume, thereby
# resulting in 10 x total no. of volumes.
# ----------------------------

# # for "Scoliotic" data
# folder_type_1 = '160'
# folder_type_2 = '23'
# patient_num = '12'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/scoliotic_patient'+patient_num+'/'
# file_name = 'normalized_patient'+patient_num+'_' + folder_type_2 + '.npy'

# # for "MICCAI" data
# mic_num = '16'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/
#              miccai_'+mic_num+'/' #+mic_num+'_wat_crpd.nrrd'
# file_name = 'normalized_'+mic_num+'_wat_crpd.npy'

# for actual training data
pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/train/A/'
load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/train/A/'
                             'patient11_160*.npy'))
# a = load_path[300].split('/')
# file_name = a[-1][:-4]
print(len(load_path))
load_path_1 = np.random.choice(load_path, 20, replace=False)   # randomly selecting 20 volumes out of 35
# print(load_path_1)
print(len(load_path_1))
# sys.exit()
mu_vals, sigma_vals = [], []
i = 0
for file in load_path_1:
    temp = file.split('/')
    file_name = temp[-1][:-4]   # gets the name of the file as it was saved except the '.npy'
    mri_data = np.load(file)

    # mri_data_norm_1 = resize(mri_data, (256, 128, 128))
    # mri_data_norm_1 = np.transpose(mri_data_norm_1, (2,1,0))
    # out = nrrd.write('/home/karthik/Desktop/p12_128.nrrd',mri_data_norm_1)
    # sys.exit()

    # aug_path = load_path
    # # randomly selecting an angle from a normal distribution with mean=0 and std=10 degrees
    # # also, randomly selecting the axes along which the 3D images will be rotated.
    # rot_xy = (1, 0)     # rotation in the XY plane
    # rot_yz = (1, 2)     # rotation in the YZ plane
    # rot_xz = (0, 2)     # rotation in the XZ plane
    # temp = [rot_xy, rot_yz, rot_xz]
    # ang, ax = [], []
    # # Use the snipped below to randomly rotate images
    # for i in range(10):
    #     angle = np.random.normal(loc=0.0, scale=10.0)
    #     rot_axes = random.choice(temp)      # even the axes of rotation are randomly chosen
    #     ang.append(angle)
    #     ax.append(rot_axes)
    #     rotated_mri_data = rotate(mri_data, angle, rot_axes, reshape=False, mode='nearest')
    #     # temp_pth = aug_path + mic_num+'_wat_crpd_aug_'+str(i)+'.npy'
    #     temp_pth = aug_path + 'patient' + patient_num + '_' + folder_type_2 + '_aug_' + str(i) + '.npy'
    #     np.save(temp_pth, rotated_mri_data)
    #     # print(rotated_mri_data.shape)

    # # Use the snippet below to add random noise
    # for i in range(10):
    #     # mu = np.random.choice(np.arange(-0.1, 0.1, 0.1))    # randomly choosing a value from -0.2 to 0.2
    #     mu = 0.0    # fixing the mean to be 0.0
    #     mu_vals.append(mu)
    #     # sigma = np.random.choice([0.01, 0.05]) # 0.1     # randomly choosing between 0.01 and 0.05 std values
    #     sigma = 0.01
    #     sigma_vals.append(sigma)
    #     noise = np.random.normal(mu, sigma, mri_data.shape)
    #     noisy_img = mri_data + noise
    #     noisy_mri_data = np.clip(noisy_img, -1, 1)
    #     temp_pth = pth + file_name[11:] + '_noise_' +str(i)+'.npy'
    #     np.save(temp_pth, noisy_mri_data)
    #     # print(noisy_mri_data.shape)
    #     # plt.figure(); plt.imshow(mri_data[:, 20, :], cmap='gray'); plt.show()
    #     # plt.figure(); plt.imshow(noisy_mri_data[:, 20, :], cmap='gray'); plt.show()

    # Use the snippet below for histogram equalization
    # hist_mri_img = equalize_hist(mri_data)    # histogram equalization
    # hist_mri_img = equalize_adapthist(mri_data)   # adaptive histogram equalization
    # contrast stretching
    p1, p99 = np.percentile(mri_data, (1, 99))
    new_mri_data = rescale_intensity(mri_data, in_range=(p1, p99))
    temp_pth = pth + file_name + '_contrastStretch_' +str(i)+'.npy'
    np.save(temp_pth, new_mri_data)
    i += 1
    # plt.figure(); plt.imshow(new_mri_data[40, :, :], cmap='gray'); plt.show()

    # # Use the snippet below for elastic deformation
    # for i in range(15):
    #     if i < 10:
    #         deformed_mri = deform_random_grid(mri_data, sigma=4.0, points=3, mode='nearest')
    #         temp_pth = pth + file_name[11:] + '_deform3x3x3_' +str(i)+'.npy'
    #         np.save(temp_pth, deformed_mri)
    #         # plt.figure(); plt.imshow(deformed_mri[:, :, 30], cmap='gray'); plt.show()
    #     else:
    #         deformed_mri = deform_random_grid(mri_data, sigma=4.0, points=5, mode='nearest')
    #         temp_pth = pth + file_name[11:] + '_deform5x5x5_' +str(i)+'.npy'
    #         np.save(temp_pth, deformed_mri)
    #         # plt.figure(); plt.imshow(deformed_mri[:, :, 30], cmap='gray'); plt.show()

# plt.figure(); plt.imshow(mri_data[:, :, 30], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(noisy_mri_data[:, 20, :], cmap='gray'); plt.show()

# # for actual CT training data
# pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/train/B/'
# # a = load_path[0].split('/')
# # file_name = a[-1][:-4]
# load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/train/B/'
#                              'normalized_cta_crp*.npy'))
# load_path_1 = np.random.choice(load_path, 15)
# # print(len(load_path_1));
# # print(load_path_1); sys.exit()
# mu_vals, sigma_vals = [], []
# i=0
# for file in load_path_1:
#     temp = file.split('/')
#     file_name = temp[-1][:-4]   # gets the name of the file as it was saved except the '.npy'
#     ct_data = np.load(file)
#
#     # mri_data_norm_1 = resize(mri_data, (256, 128, 128))
#     # mri_data_norm_1 = np.transpose(mri_data_norm_1, (2,1,0))
#     # out = nrrd.write('/home/karthik/Desktop/p12_128.nrrd',mri_data_norm_1)
#     # sys.exit()
#
#     # # aug_path = load_path
#     # # randomly selecting an angle from a normal distribution with mean=0 and std=10 degrees
#     # # also, randomly selecting the axes along which the 3D images will be rotated.
#     # rot_xy = (1, 0)     # rotation in the XY plane
#     # rot_yz = (1, 2)     # rotation in the YZ plane
#     # rot_xz = (0, 2)     # rotation in the XZ plane
#     # temp = [rot_xy, rot_yz, rot_xz]
#     # ang, ax = [], []
#     # # Use the snipped below to randomly rotate images
#     # for i in range(10):
#     #     angle = np.random.normal(loc=0.0, scale=10.0)
#     #     rot_axes = random.choice(temp)      # even the axes of rotation are randomly chosen
#     #     ang.append(angle)
#     #     ax.append(rot_axes)
#     #     rotated_ct_data = rotate(ct_data, angle, rot_axes, reshape=False, mode='nearest')
#     #     temp_pth = pth + file_name + '_aug_' + str(i) + '.npy'
#     #     np.save(temp_pth, rotated_ct_data)
#     #     # print(rotated_mri_data.shape)
#
#     # # Use the snippet below to add random noise
#     # for i in range(10):
#     #     # mu = np.random.choice(np.arange(-0.1, 0.1, 0.1))    # randomly choosing a value from -0.1 to 0.1
#     #     mu = 0.0
#     #     mu_vals.append(mu)
#     #     # sigma = np.random.choice([0.01, 0.03]) # 0.1     # keeping the standard deviation constant
#     #     sigma = 0.01
#     #     sigma_vals.append(sigma)
#     #     noise = np.random.normal(mu, sigma, ct_data.shape)
#     #     noisy_img = ct_data + noise
#     #     noisy_ct_data = np.clip(noisy_img, -1, 1)
#     #     temp_pth = pth + file_name + '_noise_' +str(i)+'.npy'
#     #     np.save(temp_pth, noisy_ct_data)
#     #     # print(rotated_mri_data.shape)
#     #     # plt.figure(); plt.imshow(ct_data[50, :, :], cmap='gray'); plt.show()
#     #     # plt.figure(); plt.imshow(noisy_ct_data[50, :, :], cmap='gray'); plt.show()
#
#     # Use the snippet below for histogram equalization
#     # hist_mri_img = equalize_hist(mri_data)    # histogram equalization
#     # hist_mri_img = equalize_adapthist(mri_data)   # adaptive histogram equalization
#     # contrast stretching
#     p1, p99 = np.percentile(ct_data, (1, 99))
#     new_ct_data = rescale_intensity(ct_data, in_range=(p1, p99))
#
#     temp_pth = pth + file_name + '_eqhist_'+str(i)+'.npy'
#     np.save(temp_pth, new_ct_data)
#     i += 1
#     # plt.figure(); plt.imshow(new_ct_data[60, :, :], cmap='gray'); plt.show()
#
#     # # Use the snippet below for elastic deformation
#     # for i in range(15):
#     #     if i < 10:
#     #         deformed_ct = deform_random_grid(ct_data, sigma=4.0, points=3, mode='nearest')
#     #         temp_pth = pth + file_name + '_deform3x3x3_' +str(i)+'.npy'
#     #         np.save(temp_pth, deformed_ct)
#     #         # plt.figure(); plt.imshow(deformed_ct[:, :, 30], cmap='gray'); plt.show()
#     #     else:
#     #         deformed_ct = deform_random_grid(ct_data, sigma=4.0, points=5, mode='nearest')
#     #         temp_pth = pth + file_name + '_deform5x5x5_' +str(i)+'.npy'
#     #         np.save(temp_pth, deformed_ct)
#     #         # plt.figure(); plt.imshow(deformed_ct[:, :, 30], cmap='gray'); plt.show()
#
# # print(ct_data.shape)
# # plt.figure(); plt.imshow(ct_data[:, :, 30]); plt.show()
# # plt.figure(); plt.imshow(rotated_ct_data[:, 40, :], cmap='gray'); plt.show()



