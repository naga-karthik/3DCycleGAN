import nrrd
import numpy as np
import glob, os, sys
import matplotlib.pyplot as plt

# # for individual files
# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/original_CT_numpy_final/norm*.npy'
# # load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_latest/test_1/A/*.npy'
# files = sorted(glob.glob(load_path))
# print(files); # sys.exit()
# # save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/testData_latest/'
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/ctData_all/'
#
# # file_name = files[0]
# for file_name in files:
#     f = file_name.split('/')
#     # print(f[-1][:-4])
#
#     # for original data files
#     ct_data = np.load(file_name) #.squeeze(0)    # loaded as: 48x256x128
#     print(ct_data.shape)#; sys.exit()
#     # for saving original data files
#     ct_data_1 = np.float32(np.transpose(ct_data, (2, 1, 0)))
#     nrrd.write(save_path + f[-1][:-4] + '.nrrd', ct_data_1)
#
#     # # for converting training generated files
#     # ct_data = np.load(file_name).squeeze(0)  # loaded as: 48x256x128
#     # ct_data_1 = np.transpose(ct_data, (0, 2, 1))
#     # nrrd.write(save_path + f[-1][:-4] + '_resized.nrrd', ct_data_1)

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


# Use the same below for testing.
# --------------------------------------- for testing .npy files --------------------------------------
epoch_nums = [30, 77, 116, 150, 199]
# epoch_nums = [70, 80, 90, 100]
epch = epoch_nums[4]
version_name = 'version8/'
fo_name = 'v8_withGC_withUnc_lgc=1.0_'+str(epch)+'/'
v_name = 'v8'     # CHANGE HERE ALSO

# load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/lightning/' \
#             'mr2ct_mcd_v4_unet_all_batchSize2_'+str(epch)+'/'
load_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_images/lightning/' \
            'five_test_patients(new)/' + version_name + fo_name
path = load_path + '*.npy'
files = sorted(glob.glob(path))

# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/lightning/' \
#             'mr2ct_mcd_v4_unet_all_batchSize2_'+str(epch)
save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/lightning/' + version_name + fo_name
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/new_test_data/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for file_name in files:
    # print(file_name)
    temp = file_name.split('/')
    name = temp[-1][:-4]
    # name = temp[-1]

    # mri_data = np.load(file_name).squeeze(0)  # loaded as: 48x256x128
    # mri_data_1 = np.transpose(mri_data, (0, 2, 1))  # because Slicer needs it as 48x128x256
    # mri_data = np.load(file_name)  # loaded as: 256x128x48
    # mri_data_1 = np.transpose(mri_data, (2, 1, 0))

    # FOR LIGHTNING-TRAINED/TESTED FILES
    mri_data = np.load(file_name)  # loaded as: 48x256x128
    mri_data_1 = np.transpose(mri_data, (0, 2, 1))  # because Slicer needs it as 48x128x256
    mri_data_2 = mri_data_1.astype('float32')   # nrrd.write only works with float32 type arrays.

    # bringing it in 0-255 range
    # mri_data_1 = (255*((mri_data_1+1)/2.0)).astype(np.uint8)  # first bringing it into 0-1 range and multiply by 255

    # nrrd.write(save_path + '/test_ep' + str(epch) + '_' + name + '.nrrd', mri_data_2)
    nrrd.write(save_path + '/' + v_name + "_epch_" + str(epch)+ "_" + name + '.nrrd', mri_data_2)

