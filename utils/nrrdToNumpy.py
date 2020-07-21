import os, sys
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
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
# folder_type_2 = '23'
# pth_mri = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/scoliotic_patient'+patient_num+'/'
# path_mri = pth_mri + '/patient'+patient_num+'_'+folder_type_2+'.nrrd'
# # mic_num = '16'
# # path_mri = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/miccai_'+mic_num+'/'+mic_num+'_wat_crpd.nrrd'
#
# mri_data, header1 = nrrd.read(path_mri, index_order='C')
# print(mri_data.shape)
#
# # mri_data = mri_data.transpose(1,2,0)    # to bring it to (256, 128, 128)
# # print(mri_data.shape)
# # plt.figure(); plt.imshow(mri_data[:, :, 1]); plt.show()
#
# # Normalize the data between -1 and 1 AND np.ptp(a) -> peak-to-peak returns nothing but np.max(a) - np.min(a)
# mri_data_norm = 2.0 * (mri_data - np.min(mri_data))/np.ptp(mri_data) - 1
# # plt.figure(); plt.imshow(mri_data_norm[40, :, :], cmap='gray'); plt.show()
#
# # for patientx_160 series & for miccai series
# if mri_data_norm.shape != (256, 128, 48):
#     mri_data_norm = resize(mri_data_norm, (256, 128, 48))   # corresponds to voxel width x height x depth
# print(mri_data_norm.shape)
# # mri_data_norm = np.transpose(mri_data_norm, (2, 1, 0))    # this transpose to used only when viewing on Slicer
#
# # # for patientx_23 series
# # if mri_data_norm.shape != (48, 256, 128):
# #     mri_data_norm = resize(mri_data_norm, (48, 256, 128))   # corresponds to voxel depth x width x height
# # print(mri_data_norm.shape)
# # # mri_data_norm = np.transpose(mri_data_norm, (0, 2, 1))      # this combination of transposing helps to view in Slicer
# # mri_data_norm = np.transpose(mri_data_norm, (1, 2, 0))      # this type of transpose is done so the dimensions match
# # print(mri_data_norm.shape)                                     # with patientx_160 series
#
# # nrrd.write('/home/karthik/Desktop/p1_48.nrrd', mri_data_norm)
# # sys.exit()
#
# # print(mri_data_norm.min(), mri_data_norm.max())
#
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/scoliotic_patient'+patient_num+'/'
# np.save(save_path+'normalized_patient'+patient_num+'_'+folder_type_2, mri_data_norm)
#
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/MRI/miccai_'+mic_num+'/'
# # np.save(save_path+'normalized_'+mic_num+'_wat_crpd', mri_data_norm)
#
#
# # Normalize the data between 0 and 1
# # mri_data_norm01 = (mri_data - np.min(mri_data))/np.ptp(mri_data)
# # print(mri_data_norm01.dtype)
# # print(mri_data_norm01.min(), mri_data_norm01.max())

# ------------- CT Data conversion -----------
# path_ct = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/CT/anon_crpd.nrrd'
pth1 = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/ground_truth_data/PATIENT11/'
gnd_ct = 'LIS_PATIENT11-label_cropped.nrrd'
pth2 = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
       'mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/epoch_200/'
seg_ct = 'test_ep200_0007_segmentation.seg.nrrd'

ct_data1, header1 = nrrd.read(pth1+gnd_ct, index_order='C')
print("Original Size: ", ct_data1.shape)
ct_data2, header2 = nrrd.read(pth2+seg_ct, index_order='C')
print("Original Size: ", ct_data2.shape)
# sys.exit()

# Normalize the data between -1 and 1 AND np.ptp(a) -> peak-to-peak returns nothing but np.max(a) - np.min(a)
# FOR GROUND TRUTH DATA
ct_data1_norm = 2.0 * (ct_data1 - np.min(ct_data1))/np.ptp(ct_data1) - 1
ct_data1_norm = np.flipud(np.transpose(ct_data1_norm, (0, 2, 1)))
print("Transposed and Flipped: ", ct_data1_norm.shape)
# plt.figure(); plt.imshow(ct_data1_norm[50, :, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(ct_data1_norm[:, 30, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(ct_data1_norm[:, :, 110], cmap='gray'); plt.show()
# # resizing it to match segmented's shape
ct_data2_resized = resize(ct_data1_norm, (220, 108, 50))
print("Resized: ", ct_data2_resized.shape)
# plt.figure(); plt.imshow(ct_data2_resized[:, :, 22], cmap='gray'); plt.show()

# because slicer needs it in 50x108x220 format, we need to convert it while saving
nrrd.write(pth1 + 'LIS_PATIENT11_resized' + '.nrrd', np.transpose(ct_data2_resized, (2, 1, 0) ))

# FOR SEGMENTED DATA
ct_data2_norm = 2.0 * (ct_data2 - np.min(ct_data2))/np.ptp(ct_data2) - 1
ct_data2_norm = np.flipud(ct_data2_norm)
print(ct_data2_norm.shape)
# plt.figure(); plt.imshow(ct_data2_norm[50, :, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(ct_data2_norm[:, 40, :], cmap='gray'); plt.show()
# plt.figure(); plt.imshow(ct_data2_norm[:, :, 20], cmap='gray'); plt.show()

sys.exit()

# The current problem is that in the ground truth data, the slice-wise segmentation covers less area compared to the
# synthesized volume segmentation. This segmentation-area-mismatch might cause large errors.

# for patientx_160 series & for miccai series
if ct_data_norm.shape != (256, 128, 48):
    ct_data_norm = resize(ct_data_norm, (256, 128, 48))   # corresponds to voxel width x height x depth
print(ct_data_norm.shape)
# ct_data_norm = np.transpose(ct_data_norm, (2, 1, 0))      # this type of transpose is done so the dimensions match
# print(ct_data_norm.shape)

# # DON'T USE THIS
# if ct_data_norm.shape != (128, 256, 48):
#     ct_data_norm = resize(ct_data_norm, (128, 256, 48))   # corresponds to voxel width x height x depth
# print(ct_data_norm.shape)
# ct_data_norm = np.transpose(ct_data_norm, (2, 0, 1))      # this type of transpose is done so the dimensions match
# print(ct_data_norm.shape)

# nrrd.write('/home/karthik/Desktop/ct_48_1.nrrd', ct_data_norm)
# sys.exit()

save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/3DcycleGAN/datasets/CT/'
np.save(save_path+'normalized_anon_crpd', ct_data_norm)
#

# ct_data, header2 = nrrd.read(path_ct, index_order='C')
# print(ct_data.shape)
# na = ct_data.shape[0]
# for i in range(na):
#     filename = ct_save_path+'/'+'axial'+str(i+1)+'.png'
#     I = ct_data[i,:,:]
#     img1 = (((I - I.min())/(I.max() - I.min())) * 255).astype(np.uint8)
#     img1 = Image.fromarray(img1).resize((200,200))
#     img1.save(filename)

# plt.figure()
# plt.imshow(ct_data[:,:,1], cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.show()


