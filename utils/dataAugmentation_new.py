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
# patient_num = '2'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_withScoliotic/train/A/'
# file_name = 'normalized_patient'+patient_num+'_' + folder_type_1 + '.npy'

# for "MICCAI" data
# mic_num = '01'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/
#              miccai_'+mic_num+'/' #+mic_num+'_wat_crpd.nrrd'
# file_name = 'normalized_'+mic_num+'_wat_crpd.npy'

# # for actual training data
# pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_newData_withScoliotic/train/A/'
# # load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
# #                              'mr2ct_newData_withScoliotic/train/A/normalized_patient*160.npy'))
# load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
#                              'mr2ct_newData_withScoliotic/train/A/normalized_*_wat_crpd.npy'))
# print(len(load_path))
# # load_path_1 = np.random.choice(load_path, 20, replace=False)   # randomly selecting 20 volumes out of 35
# # print(len(load_path_1))
# # sys.exit()
# i = 0
# for file in load_path:
#     temp = file.split('/')
#     file_name = temp[-1][:-4]   # gets the name of the file as it was saved except the '.npy'
#     mri_data = np.load(file)
#
#     # # randomly selecting an angle from a normal distribution with mean=0 and std=10 degrees
#     # # also, randomly selecting the axes along which the 3D images will be rotated.
#     # rot_xy = (1, 0)     # rotation in the XY plane
#     # rot_yz = (1, 2)     # rotation in the YZ plane
#     # rot_xz = (0, 2)     # rotation in the XZ plane
#     # temp = [rot_xy, rot_yz, rot_xz]
#     # ang, ax = [], []
#     # # Use the snipped below to randomly rotate images
#     # for i in range(15):
#     #     angle = np.random.normal(loc=0.0, scale=10.0)
#     #     rot_axes = random.choice(temp)      # even the axes of rotation are randomly chosen
#     #     ang.append(angle)
#     #     ax.append(rot_axes)
#     #     rotated_mri_data = rotate(mri_data, angle, rot_axes, reshape=False, mode='nearest')
#     #     # temp_pth = aug_path + mic_num+'_wat_crpd_aug_'+str(i)+'.npy'
#     #     temp_pth = pth + file_name + '_rotated_' + str(i) + '.npy'
#     #     np.save(temp_pth, rotated_mri_data)
#     #     # print(rotated_mri_data.shape)
#     # # print(ang, ax)
#     # # sys.exit()
#
#     # # Use the snippet below to add random noise
#     # for i in range(15):
#     #     mu, sigma = 0.0, 0.01
#     #     noise = np.random.normal(mu, sigma, mri_data.shape)
#     #     noisy_img = mri_data + noise
#     #     noisy_mri_data = np.float32(np.clip(noisy_img, -1, 1))
#     #     temp_pth = pth + file_name + '_noise_' +str(i)+'.npy'
#     #     np.save(temp_pth, noisy_mri_data)
#     #     # print(noisy_mri_data.shape)
#     #     # plt.figure(); plt.imshow(mri_data[:, 20, :], cmap='gray'); plt.show()
#     #     # plt.figure(); plt.imshow(noisy_mri_data[:, 20, :], cmap='gray'); plt.show()
#     # # sys.exit()
#
#     # # Use the snippet below for histogram equalization
#     # # contrast stretching
#     # p1, p99 = np.percentile(mri_data, (1, 99))
#     # new_mri_data = rescale_intensity(mri_data, in_range=(p1, p99))
#     # temp_pth = pth + file_name + '_contrastStretch_' +str(i)+'.npy'
#     # np.save(temp_pth, new_mri_data)
#     # i += 1
#     # # plt.figure(); plt.imshow(new_mri_data[40, :, :], cmap='gray'); plt.show()
#
#     # Use the snippet below for elastic deformation
#     for i in range(15):
#         if i < 10:
#             deformed_mri = deform_random_grid(mri_data, sigma=4.0, points=3, mode='nearest')
#             temp_pth = pth + file_name + '_deform3x3x3_' +str(i)+'.npy'
#             np.save(temp_pth, deformed_mri)
#             # plt.figure(); plt.imshow(deformed_mri[:, :, 30], cmap='gray'); plt.show()
#         else:
#             deformed_mri = deform_random_grid(mri_data, sigma=4.0, points=5, mode='nearest')
#             temp_pth = pth + file_name + '_deform5x5x5_' +str(i)+'.npy'
#             np.save(temp_pth, deformed_mri)
#             # plt.figure(); plt.imshow(deformed_mri[:, :, 30], cmap='gray'); plt.show()
#     # sys.exit()

# plt.figure(); plt.imshow(mri_data[:, :, 30], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(noisy_mri_data[:, 20, :], cmap='gray'); plt.show()

# --------------------------------------------
# --- for actual CT training data ----
pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_newData_withScoliotic/train/B/'
load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
                             'mr2ct_newData_withScoliotic/train/B/*crpd.npy'))
# pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/fixed_new_ct_data_numpy/'
# load_path = sorted(glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/fixed_new_ct_data_numpy/'
#                              '*crpd.npy'))
# print(len(load_path)); sys.exit()
load_path_1 = np.random.choice(load_path, 9, replace=False)
# print(load_path_1); sys.exit()
i=0
for file in load_path_1:
    temp = file.split('/')
    file_name = temp[-1][:-4]   # gets the name of the file as it was saved except the '.npy'
    ct_data = np.load(file)

    # data = np.float32(ct_data)
    # temp_pth = pth + file_name + '.npy'
    # np.save(temp_pth, data)

    # # randomly selecting an angle from a normal distribution with mean=0 and std=10 degrees
    # # also, randomly selecting the axes along which the 3D images will be rotated.
    # rot_xy = (1, 0)     # rotation in the XY plane
    # rot_yz = (1, 2)     # rotation in the YZ plane
    # rot_xz = (0, 2)     # rotation in the XZ plane
    # temp = [rot_xy, rot_yz, rot_xz]
    # ang, ax = [], []
    # # Use the snipped below to randomly rotate images
    # for i in range(1):
    #     angle = np.random.normal(loc=0.0, scale=10.0)
    #     rot_axes = random.choice(temp)      # even the axes of rotation are randomly chosen
    #     ang.append(angle)
    #     ax.append(rot_axes)
    #     rotated_ct_data = rotate(ct_data, angle, rot_axes, reshape=False, mode='nearest')
    #     temp_pth = pth + file_name + '_rotated_' + str(i) + '.npy'
    #     np.save(temp_pth, rotated_ct_data)
    #     # print(rotated_ct_data.shape)
    # # print(ang, ax)
    # # sys.exit()

    # # Use the snippet below to add random noise
    # for i in range(1):
    #     mu, sigma = 0.0, 0.01
    #     noise = np.random.normal(mu, sigma, ct_data.shape)
    #     noisy_img = ct_data + noise
    #     noisy_ct_data = np.float32(np.clip(noisy_img, -1, 1))
    #     temp_pth = pth + file_name + '_noise_' +str(i)+'.npy'
    #     np.save(temp_pth, noisy_ct_data)
    #     # print(rotated_mri_data.shape)
    #     # plt.figure(); plt.imshow(ct_data[50, :, :], cmap='gray'); plt.show()
    #     # plt.figure(); plt.imshow(noisy_ct_data[50, :, :], cmap='gray'); plt.show()
    # # sys.exit()

    # Use the snippet below for elastic deformation
    for i in range(1):
        if i < 1:
            deformed_ct = deform_random_grid(ct_data, sigma=4.0, points=3, mode='nearest')
            temp_pth = pth + file_name + '_deform3x3x3_' +str(i)+'.npy'
            np.save(temp_pth, deformed_ct)
            # plt.figure(); plt.imshow(deformed_ct[:, :, 30], cmap='gray'); plt.show()
        else:
            deformed_ct = deform_random_grid(ct_data, sigma=4.0, points=5, mode='nearest')
            temp_pth = pth + file_name + '_deform5x5x5_' +str(i)+'.npy'
            np.save(temp_pth, deformed_ct)
            # plt.figure(); plt.imshow(deformed_ct[:, :, 30], cmap='gray'); plt.show()
    # sys.exit()

# # # print(ct_data.shape)
# # # plt.figure(); plt.imshow(ct_data[:, :, 30]); plt.show()
# # # plt.figure(); plt.imshow(rotated_ct_data[:, 40, :], cmap='gray'); plt.show()



