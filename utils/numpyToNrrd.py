import nrrd
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt

# # --------------- for training .npy files ----------------
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated/' \
#             'mr2ct_new_unetUpdated/realA2fakeB/'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated' \
# #             '/mr2ct_unet_batchsize16/realA2fakeB/'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/training_generated/
# #              mr2ct_spectral/realA2fakeB/'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct/test/A/
# #              normalized_patient4_160.npy'
#
# path = load_path+'*.npy'
# files = sorted(glob.glob(path))
#
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_new_unetUpdated'
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_unet'
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_spectral'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# file_name = files[15]
# f = file_name.split('/')
# print(f[-1][:-4])
# mri_data = np.load(file_name).squeeze(0)    # loaded as: 48x256x128
# print(mri_data.shape, mri_data.dtype, mri_data.min(), mri_data.max())
# # sys.exit()
# mri_data_1 = np.transpose(mri_data, (0, 2, 1))
# # bringing it in 0-255 range
# # mri_data_1 = (255*((mri_data_1+1)/2.0)).astype(np.uint8)   # first bringing it into 0-1 range and multipyling my 255
# nrrd.write(save_path+'/train_' + f[-1][:-4] + '.nrrd', mri_data_1)
# # nrrd.write(save_path+'/mr_original.nrrd', mri_data_1)


# --------------- for testing .npy files ----------------
# epoch_nums = [40, 80, 120, 160, 200]
epoch_nums = [70, 80, 90, 100]
epch = epoch_nums[3]
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/mr2ct_spectral' \
#             '/realA2fakeB/epoch_' + str(epch) + '/'
load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/testing_generated/' \
            'mr2ct_new_unetUpdated/realA2fakeB/epoch_' + str(epch) + '/'
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct/test_diverse/A/'
path = load_path + '*.npy'
files = sorted(glob.glob(path))

# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
#             'mr2ct_unet_gradient_consistency/diverse_epoch_' + str(epch)
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_unet/diverse_epoch_'+str(epch)
save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/mr2ct_new_unetUpdated/epoch_'+str(epch)
if not os.path.exists(save_path):
    os.mkdir(save_path)

for file_name in files:
    temp = file_name.split('/')
    name = temp[-1][:-4]
    mri_data = np.load(file_name).squeeze(0)  # loaded as: 48x256x128
    mri_data_1 = np.transpose(mri_data, (0, 2, 1))
    # bringing it in 0-255 range
    # mri_data_1 = (255*((mri_data_1+1)/2.0)).astype(np.uint8)  # first bringing it into 0-1 range and multiply by 255
    # nrrd.write(save_path+'/grad_con.nrrd', mri_data_1)
    nrrd.write(save_path + '/test_newUpdated_' + name + '.nrrd', mri_data_1)

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
