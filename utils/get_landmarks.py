import numpy as np
import sys

# ----------- Landmarks points ----------------
# this getting the landmarks from .o3 file for each patient.
# dir_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/ANONwithDICOM/'
dir_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/ANON_latest_5Patients/'
patient_num = 'PATIENT3/'
date = '2007-11-14/' # '2008-03-20/'  # '2007-07-25/' # '2007-04-10/' # '2007-11-06/'
# file_name = 'LIS_PATIENT12_edited.o3'
# file_name = 'LIS_PATIENT11_edited_thoracic.o3'
# file_name = 'LIS_PATIENT1_edited.o3'
file_name = 'LIS_PATIENT3_edited.o3'
# file_name = 'LIS_PATIENT4_edited.o3'
file = dir_path+patient_num+date+file_name
save_folder_name = 'individual_vert_p3_vol07/'
p_num = 'p3'

vname = 'vert_T12'
with open(file, 'rt') as f_:
    lines = f_.readlines()
landmarks = {}
for i in range(len(lines)):
    x_coord, y_coord, z_coord = [], [], []
    if lines[i].startswith("Objet"):
        name = lines[i].split()[1]
        if name == 'Vertebre_T12':      # each vertebra has 74 points, except for T12 which has 70 points
            # change name here and also at the bottom where file is saved.
            for j in range(2, 19):  # earlier it was (3, 76) and first line was missed. for T12 use (2, 72)
                                    # PATIENT3 has only 17 points unlike others -> so use (2, 19)
                if not lines[i+j].startswith("#"):  # ignore lines that start with '#'
                    data = lines[i+j].split("   ")
                    # print(data, len(data))
                    # sys.exit()
                    if len(data) == 5:      # for PATIENT3 use 7
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[3]))
                        z_coord.append(float(data[4].rstrip(" \n")))
                    else:
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[2]))
                        z_coord.append(float(data[3].rstrip(" \n")))
                else:
                    continue
            landmarks.update({name: (x_coord, y_coord, z_coord)})
            # result = np.array([[x_coord], [y_coord], [z_coord]])
            # result = np.array([x_coord, y_coord, z_coord])
        else:
            continue
        # else:
        #     for j in range(3, 72):
        #         if not lines[i+j].startswith("#"):
        #             data = lines[i+j].split("   ")
        #             # print(data, len(data))
        #             # sys.exit()
        #             if len(data) == 5:
        #                 x_coord.append(float(data[1]))
        #                 y_coord.append(float(data[3]))
        #                 z_coord.append(-1.0*float(data[4].rstrip(" \n")))
        #             else:
        #                 x_coord.append(float(data[1]))
        #                 y_coord.append(float(data[2]))
        #                 z_coord.append(-1.0*float(data[3].rstrip(" \n")))
        #         else:
        #             continue
        #     landmarks.update({ name: (x_coord, y_coord, z_coord)})

xdata, ydata, zdata = [], [], []
for k, v in landmarks.items():
    xdata.append(v[0])
    ydata.append(v[1])
    zdata.append(v[2])
# # print(xdata)
array = np.array([xdata, ydata, zdata])     # shape = 73x1x3
array = array.squeeze()      # shape = 3x73
print(array.shape)

# saving the numpy array
# save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
#             'mr2ct_v4_lighterUnet_gradientConsistency_batchSize4/epoch_200/individual_vertebrae_vol_0006/'
save_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/nrrd_files/' \
            'lightning/landmark_points/'+save_folder_name
np.save(save_path+ 'original_' + p_num + '_' + vname + '_coords', array)