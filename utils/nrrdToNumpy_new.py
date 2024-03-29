import os, sys
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from PIL import Image
import glob

# A Summary of what this file does:
# Firstly, before using this code, the DICOM images are loaded in Slicer3D and the volume is cropped. The cropped
# volume is saved as a .nrrd file. "That" file is loaded here and resized to a volume of size 256 x 128 x 48. Note
# that the original .nrrd file has data in uint16 format. The data is normalized to lie between -1 and 1, which
# has converted the data to float64 format.
# --------------------------------

# patient_num= '12'
# folder_type_1 = '160'
# pth_mri = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_MR/'
# path_mri = pth_mri + '/patient'+patient_num+'_'+folder_type_1+'.nrrd'
# # mic_num = '16'
# # path_mri = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_MR/'+mic_num+'_wat_crpd.nrrd'
#
# mri_data, header1 = nrrd.read(path_mri, index_order='C')
# print(mri_data.shape)
#
# # Normalize the data between -1 and 1 AND np.ptp(a) -> peak-to-peak returns nothing but np.max(a) - np.min(a)
# mri_data_norm = 2.0 * (mri_data - np.min(mri_data))/np.ptp(mri_data) - 1
# # plt.figure(); plt.imshow(mri_data_norm[40, :, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(mri_data_norm[:, 40, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(mri_data_norm[:, :, 20], cmap='gray'); plt.show()
#
# # mri_data_norm = np.transpose(mri_data_norm, (1,2,0))    # use this only for patient6
#
# # for patientx_160 series & for miccai series
# if mri_data_norm.shape != (256, 128, 48):
#     # # for miccai
#     # mri_data_norm = np.fliplr(resize(mri_data_norm, (256, 128, 48)))  # corresponds to voxel width x height x depth
#     # # the above type of flipping works for MICCAI
#
#     # for scoliotic
#     # mri_data_norm = np.flipud(resize(mri_data_norm, (256, 128, 48)))  # corresponds to voxel width x height x depth
#     # #this (above) was how the data was fed to the model originally. it turns that it is a flipped version and FAIlS.
#     mri_data_norm = (resize(mri_data_norm, (256, 128, 48)))  # corresponds to voxel width x height x depth
#     # #this (above) was how test data was fixed, it shows the correct view (not flipped)
#     # mri_data_norm = np.flip(resize(np.transpose(mri_data_norm, (1, 2, 0)), (256, 128, 48)))   # use only for patient6
# print(mri_data_norm.shape)
# # mri_data_norm = np.transpose(mri_data_norm, (2, 1, 0))    # this transpose to used only when viewing on Slicer
#
# # contrast stretching
# p1, p99 = np.percentile(mri_data_norm, (0.1, 99.9))
# mri_data_norm_1 = np.float32(rescale_intensity(mri_data_norm, in_range=(p1, p99)))
#
# # plt.figure(); plt.imshow(mri_data_norm_1[40, :, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(mri_data_norm_1[:, 60, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(mri_data_norm_1[:, :, 15], cmap='gray'); plt.show()
# # sys.exit()
#
# # saving path for scoliotic
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_MR_numpy/'
# np.save(save_path+'normalized_patient'+patient_num+'_'+folder_type_1, mri_data_norm_1)
# # # saving path for miccai
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_MR_numpy/'
# # np.save(save_path+'normalized_'+mic_num+'_wat_crpd', mri_data_norm_1)
#

# # Lists for 'spine-1' folder
# patients_list = ['0016', '0017', '0023', '0027', '0030', '0035', '0037', '0039', '0045', '0047', '0059', '0068', '0077', '0078']
# ids_list = ['4564449', '4542898', '4542094', '4580220', '4562828', '4542183', '4574394', '4533045', '2504978', '2754140', '2539767', '3164509', '4533772', '2551924']
# # Lists for 'spine-2' folder
# patients_list = ['0105', '0108', '0109', '0112', '0122', '0124', '0126', '0128', '0140', '0144', '0146', '0153']
# ids_list = ['4526976', '4552200', '4522160', '4568017', '4557138', '4573639', '4553119', '4536524', '3023828', '4574645', '4568220', '4516999']
# Lists for 'spine-5' folder
# patients_list = ['0385', '0396', '0419', '0428', '0437', '0439', '0452', '0453', '0455', '0463', '0479', '0486', '0499']
# ids_list = ['4535237', '4545016', '2947667', '4567263', '4534523', '2665648', '4569602', '4559282', '4482572', '4540562', '4575448', '4515603', '4407253']
# patients_list_1 = ['0499']
# ids_list_1 = ['4407254']

load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_CT/*.nrrd'
files = sorted(glob.glob(load_path))
files = files[:5]
# print(len(files))
# sys.exit()

# # ------------- CT Data conversion for multiple files -----------
# pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/spine-5/'
# for p_num, p_id in zip(patients_list_1, ids_list_1):
#     file_name = 'patient'+p_num+'/'+p_id+'_cropped.nrrd'
#     path_ct = pth+file_name
#     ct_data1, header1 = nrrd.read(path_ct, index_order='C')
#     print("Original Size: ", ct_data1.shape)
#     # sys.exit()

# A few CT data saved from Lab PC use a different version of numpy (and there is a different version in the laptop)
# That has caused problems while loading the data. Hence the data which was saved using Lab PC are copied here, re-saved
# re-augmented and put back into the main data.
for file in files:
    ct_data1, header1 = nrrd.read(file, index_order='C')
    print("Original Size: ", ct_data1.shape)
    # sys.exit()

    # Normalize the data between -1 and 1 AND np.ptp(a) -> peak-to-peak returns nothing but np.max(a) - np.min(a)
    # FOR GROUND TRUTH DATA
    ct_data1_norm = 2.0 * (ct_data1 - np.min(ct_data1))/np.ptp(ct_data1) - 1
    # ct_data1_norm = np.flipud(np.transpose(ct_data1_norm, (0, 1, 2)))     # use np.flip here
    ct_data1_norm = (np.transpose(ct_data1_norm, (0, 1, 2)))
    print("Transposed and Flipped: ", ct_data1_norm.shape)
    # plt.figure(); plt.imshow(ct_data1_norm[50, :, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(ct_data1_norm[:, 50, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(np.fliplr(ct_data1_norm[:, :, 30]), cmap='gray'); plt.show()
    # sys.exit()

    if ct_data1_norm.shape != (256, 128, 48):
        ct_data_norm = resize(ct_data1_norm, (256, 128, 48))   # corresponds to voxel width x height x depth
    ct_data_norm = np.float32(ct_data_norm)
    print(ct_data_norm.shape, ct_data_norm.dtype)
    # ct_data_norm = np.transpose(ct_data_norm, (2, 1, 0))      # this type of transpose is done so the dimensions match
    # print(ct_data_norm.shape)
    # plt.figure(); plt.imshow(ct_data_norm[50, :, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(ct_data_norm[:, 50, :], cmap='gray'); plt.show()
    # plt.figure(); plt.imshow(ct_data_norm[:, :, 30], cmap='gray'); plt.show()
    # sys.exit()

    # contrast stretching
    p1, p99 = np.percentile(ct_data_norm, (0.5, 99.5))
    new_ct_data = rescale_intensity(ct_data_norm, in_range=(p1, p99))

    # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_CT_numpy/'
    save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
    f = file.split('/')
    f1 = f[-1][:-13]
    np.save(save_path+ f1 +'_crpd', new_ct_data)

# # ------------- CT Data conversion for single files -----------
# pth = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/normalized_cta_crpd.npy'
# ct_data = np.load(pth)
# new_ct_data = np.float32(ct_data)
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
# np.save(save_path + 'normalized_cta_crpd_1', new_ct_data)
# sys.exit()
#
# ct_data1, header1 = nrrd.read(pth, index_order='C')
# print("Original Size: ", ct_data1.shape)
# # sys.exit()
#
# # Normalize the data between -1 and 1 AND np.ptp(a) -> peak-to-peak returns nothing but np.max(a) - np.min(a)
# # FOR GROUND TRUTH DATA
# ct_data1_norm = 2.0 * (ct_data1 - np.min(ct_data1))/np.ptp(ct_data1) - 1
# ct_data1_norm = np.flipud(np.transpose(ct_data1_norm, (0, 1, 2)))
# print("Transposed and Flipped: ", ct_data1_norm.shape)
# # plt.figure(); plt.imshow(ct_data1_norm[50, :, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(ct_data1_norm[:, 50, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(np.fliplr(ct_data1_norm[:, :, 30]), cmap='gray'); plt.show()
# # sys.exit()
#
# if ct_data1_norm.shape != (256, 128, 48):
#     ct_data_norm = resize(ct_data1_norm, (256, 128, 48))   # corresponds to voxel width x height x depth
# ct_data_norm = np.float32(ct_data_norm)
# print(ct_data_norm.shape, ct_data_norm.dtype)
# # ct_data_norm = np.transpose(ct_data_norm, (2, 1, 0))      # this type of transpose is done so the dimensions match
# # print(ct_data_norm.shape)
# # plt.figure(); plt.imshow(ct_data_norm[50, :, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(ct_data_norm[:, 50, :], cmap='gray'); plt.show()
# # plt.figure(); plt.imshow(ct_data_norm[:, :, 30], cmap='gray'); plt.show()
# # sys.exit()
#
# # contrast stretching
# p1, p99 = np.percentile(ct_data_norm, (0.1, 99.9))
# new_ct_data = rescale_intensity(ct_data_norm, in_range=(p1, p99))
#
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/'
# np.save(save_path + 'cta_crpd_resized', new_ct_data)


