import nrrd
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt

# # --------------- for training .npy files ----------------
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated/' \
# #             'mr2ct_new_unetUpdated/realA2fakeB/'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated/' \
#             'mr2ct_v3_lighterUnet_gradientConsistency_batchSize4/realA2fakeB/0001.npy'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated/' \
# #             'mr2ct_spectral/realA2fakeB/'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/test/A/normalized_patient11_23.npy'
# files = sorted(glob.glob(load_path))
# # path = load_path+'*.npy'
# # files = sorted(glob.glob(path))
#
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_new_unetUpdated'
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_unet'
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
#             'mr2ct_v3_lighterUnet_gradientConsistency_batchSize4'
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/test/A1'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# file_name = files[0]
# f = file_name.split('/')
# # print(f[-1][:-4])
# mri_data = np.load(file_name).squeeze(0)    # loaded as: 48x256x128
# print(mri_data.shape, mri_data.dtype, mri_data.min(), mri_data.max())
# # sys.exit()
# mri_data_1 = np.transpose(mri_data, (0, 2, 1))
#
# # mri_data_1 = np.transpose(mri_data, (2, 1, 0))
# # mri_data_1 = np.flip(mri_data) #, axis=1)
# # print(mri_data_1.shape)
# # np.save(save_path+'/' + f[-1][:-4], mri_data_1)
#
# # bringing it in 0-255 range
# # mri_data_1 = (255*((mri_data_1+1)/2.0)).astype(np.uint8)   # first bringing it into 0-1 range and multipyling my 255
#
# nrrd.write(save_path+'/train_' + f[-1][:-4] + '.nrrd', mri_data_1)
# # nrrd.write(save_path+'/' + f[-1][:-4] + '.nrrd', mri_data_1)


# # --------------- for testing .npy files ----------------
# epoch_nums = [60, 80, 120, 160, 180]
# # epoch_nums = [70, 80, 90, 100]
# epch = epoch_nums[4]
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spectral' \
# #             '/realA2fakeB/epoch_' + str(epch) + '/'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/' \
# #             'mr2ct_unet_gradcon_exp1/realA2fakeB/epoch_' + str(epch) + '/'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/' \
#             'mr2ct_v1_lighterUnet_gradientConsistency_batchSize4/realA2fakeB/epoch_' + str(epch) + '/'
# path = load_path + '*.npy'
# files = sorted(glob.glob(path))
#
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
# #             'mr2ct_unet_gradient_consistency/diverse_epoch_' + str(epch)
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_unet/diverse_epoch_'+str(epch)
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
#             'mr2ct_v1_lighterUnet_gradientConsistency_batchSize4/epoch_' +str(epch)
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/new_test_data/'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
#
# # print(np.load(files[0]).shape)
# # sys.exit()
# for file_name in files:
#     temp = file_name.split('/')
#     name = temp[-1][:-4]
#     mri_data = np.load(file_name).squeeze(0)  # loaded as: 48x256x128
#     mri_data_1 = np.transpose(mri_data, (0, 2, 1))  # because Slicer needs it as 48x128x256
#     # mri_data = np.load(file_name)  # loaded as: 256x128x48
#     # mri_data_1 = np.transpose(mri_data, (2, 1, 0))
#
#     # bringing it in 0-255 range
#     # mri_data_1 = (255*((mri_data_1+1)/2.0)).astype(np.uint8)  # first bringing it into 0-1 range and multiply by 255
#
#     # nrrd.write(save_path+'/gradcon_exp1_new_data_' + name + '.nrrd', mri_data_1)
#     nrrd.write(save_path + '/test_ep180_' + name + '.nrrd', mri_data_1)

# -------------------------------------------------------------------------
# shape of the original 3D matrix MRI - (217, 123, 104)
# shape of the original 3D matrix CT - (246, 130, 112)
# shape of the original axial slice - (123, 104)
# shape of the original sagittal slice - (217, 123)
# shape of the original coronal slice - (217, 104)

# final_mat = np.zeros((217, 123, 104))
#
# axial_paths = glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN/testing_generated/mr2ct/mr2ct_axial'
#                         '/realA2fakeB/*.*')
# n = len(axial_paths)
# # axial = np.zeros((n,104,123))
# axial = []
# for i, paths in enumerate(axial_paths):
#     img = Image.open(paths)
#     img = img.resize((123, 104)).convert("L")
#     img = np.asarray(img).T
#     # axial.append(img)
#     final_mat[i, :, :] = img

# print(len(axial))
# ax = np.array(axial)
# print(ax.shape)

# plt.imshow(ax[:,:,1])
# plt.show()
# save_path = '/home/karthik/PycharmProjects/SlicerExperiments/nrrdFiles/'
# nrrd.write(save_path+'converted_axial.nrrd', axial)

# sag_path = glob.glob('/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN/testing_generated/mr2ct/mr2ct_sagittal'
#                      '/realA2fakeB/*.*')
# n2 = len(sag_path)
# # sagittal = np.zeros((n1, 123, 217))
# sagittal = []
# for i, paths in enumerate(sag_path):
#     img = Image.open(paths)
#     img = img.resize((217, 123)).convert("L")
#     img = np.asarray(img).T
#     # sagittal.append(np.asarray(img).T)
#     final_mat[:, :, i] = img

# print(len(sagittal))
# sa = np.array(sagittal)
# print(sa.shape)

# save_path = '/home/karthik/PycharmProjects/SlicerExperiments/nrrdFiles/'
# nrrd.write(save_path+'converted_sagittal.nrrd', sagittal)

# plt.figure(); plt.imshow(final_mat[1,:,:]); plt.show()
# plt.figure(); plt.imshow(final_mat[:,:,1]); plt.show()
# print(final_mat[1,:,:])
# print(final_mat[:,:,1])
# print(final_mat[1,:,:] == final_mat[:,:,1])
