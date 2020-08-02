import sys
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
plotly.offline.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# IN THIS FILE, THE POINTS ARE NOT NORMALIZED. THEY ARE USED AS IT IS, COMPUTING THE CENTROID, CALCULATING THE
# EIGENVECTORS AND ALIGNING THEM ONE-BY-ONE.

# ----------- Landmarks points ----------------
# this getting the landmarks from .o3 file for each patient.
dir_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/ANONwithDICOM/PATIENT11/2007-07-25/'
file_name = 'LIS_PATIENT11_edited_thoracic.o3'
# file_name = 'LIS_PATIENT11_edited_thoracic_1.o3'
file = dir_path+file_name

with open(file, 'rt') as f_:
    lines = f_.readlines()
landmarks = {}
for i in range(len(lines)):
    x_coord, y_coord, z_coord = [], [], []
    if lines[i].startswith("Objet"):
        name = lines[i].split()[1]
        if name == 'Vertebre_T7':      # each vertebra has 73 points, except for T12 which has 69 points
            for j in range(3, 76):
                if not lines[i+j].startswith("#"):  # ignore lines that start with '#'
                    data = lines[i+j].split("   ")
                    # print(data, len(data))
                    # sys.exit()
                    if len(data) == 5:
                        # those numbers are translation factors for each axis. This is to specifically align the mean/
                        # centroid of the segmented and the landmark points.
                        x_coord.append(float(data[1]) + 45.27613014)
                        y_coord.append(float(data[3]) - 24.5141186)
                        z_coord.append(float(data[4].rstrip(" \n")) - 343.8056481)

                        # x_coord.append(float(data[1]))
                        # y_coord.append(float(data[3]))
                        # z_coord.append(float(data[4].rstrip(" \n")))

                    # the z-coordinates are negated because the segmented images are the mirror images of the original
                    # images (that is how the model generated the synthesized images)
                    else:
                        x_coord.append(float(data[1]) + 45.27613014)
                        y_coord.append(float(data[2]) - 24.5141186)
                        z_coord.append(float(data[3].rstrip(" \n")) - 343.8056481)

                        # x_coord.append(float(data[1]))
                        # y_coord.append(float(data[2]))
                        # z_coord.append(float(data[3].rstrip(" \n")))
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
# print(array.shape)

# computing the 3-dimensional mean vector
land_mean_x = np.mean(array[0,:])
land_mean_y = np.mean(array[1,:])
land_mean_z = np.mean(array[2,:])
land_mean_vec = np.array([[land_mean_x], [land_mean_y], [land_mean_z]])     # 3x1
print("Centroid Coordinates: \n", land_mean_vec)

scatter_matrix_landmarks = np.zeros((3,3))
for i in range(array.shape[1]):
    scatter_matrix_landmarks += (array[:,i].reshape(3,1) - land_mean_vec).dot((array[:,i].reshape(3,1) - land_mean_vec).T)
print('Landmarks Scatter Matrix:\n', scatter_matrix_landmarks) #; sys.exit()

# eigenvalues and eigenvectors of the scatter matrix
land_eig_val_sc, land_eig_vec_sc = np.linalg.eig(scatter_matrix_landmarks)
# print("Eigenvalues from scatter matrix: \n", land_eig_val_sc)
# print("Scatter Matrix Eigenvectors: \n", land_eig_vec_sc)
# print("Scatter Matrix Eigenvectors: \n", land_eig_vec_sc.T)
# print("Eigen: \n", land_eig_vec_sc.T[1])

# arranging the eigenvalues and eigenvectors in the descending order.
# making a list of eigenvalues, eigenvector tuples
land_eig_pairs = [(np.abs(land_eig_val_sc[i]), land_eig_vec_sc[:,i]) for i in range(len(land_eig_vec_sc))]
land_eig_pairs.sort(key=lambda x: x[0], reverse=True)
desc_land_eig_vec = [np.hstack((land_eig_pairs[i][1].reshape(3,1))) for i in range(3)]
desc_land_eig_vec = (np.array(desc_land_eig_vec)).T
print("All Eigenvectors: \n", desc_land_eig_vec)

# ---------------------- Segmented Points ---------------------------
path = './vol7_pat11_vertT7_coords.npy'
vol_data = np.load(path)
vol_data = vol_data.T

# computing the 3-dimensional mean vector
seg_mean_x = np.mean(vol_data[0,:])
seg_mean_y = np.mean(vol_data[1,:])
seg_mean_z = np.mean(vol_data[2,:])
seg_mean_vec = np.array([[seg_mean_x], [seg_mean_y], [seg_mean_z]])     # 3x1
print("Mean per dimension: \n", seg_mean_vec) #; sys.exit()

scatter_matrix_segment = np.zeros((3,3))
for i in range(vol_data.shape[1]):
    scatter_matrix_segment += (vol_data[:,i].reshape(3,1) - seg_mean_vec).dot((vol_data[:,i].reshape(3,1) - seg_mean_vec).T)
print('Segmentation Scatter Matrix:\n', scatter_matrix_segment) #; sys.exit()

# eigenvalues and eigenvectors of the scatter matrix
seg_eig_val_sc, seg_eig_vec_sc = np.linalg.eig(scatter_matrix_segment)
# print("Eigenvalues from scatter matrix: \n", seg_eig_val_sc)
# print("Scatter Matrix Eigenvectors' shape: \n", seg_eig_vec_sc.shape)

# arranging the eigenvalues and eigenvectors in the descending order.
# making a list of eigenvalues, eigenvector tuples
seg_eig_pairs = [(np.abs(seg_eig_val_sc[i]), seg_eig_vec_sc[:,i]) for i in range(len(seg_eig_vec_sc))]
seg_eig_pairs.sort(key=lambda x: x[0], reverse=True)
desc_seg_eig_vec = [np.hstack((seg_eig_pairs[i][1].reshape(3,1))) for i in range(3)]
desc_seg_eig_vec = (np.array(desc_seg_eig_vec)).T
# print("All Eigenvectors: \n", desc_seg_eig_vec)
# print("Largest Eigenvector: \n", desc_seg_eig_vec[:1])
# print("Second Eigenvector: \n", desc_seg_eig_vec[1:2])
# print("Last Eigenvector: \n", desc_seg_eig_vec[2:3])
# sys.exit()


def x_rotation(vector, theta):
    theta = np.radians(theta)
    R_yz = np.array(((1, 0, 0), (0, np.cos(theta), -np.sin(theta)), (0, np.sin(theta), np.cos(theta))))
    return np.dot(R_yz, vector)


def y_rotation(vector, theta):
    theta = np.radians(theta)
    R_xz = np.array(((np.cos(theta), 0, np.sin(theta)), (0, 1, 0), (-np.sin(theta), 0, np.cos(theta))))
    return np.dot(R_xz, vector)


def z_rotation(vector, theta):
    theta = np.radians(theta)
    R_xy = np.array(((np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0, 1)))
    return np.dot(R_xy, vector)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # plot landmark points
# # ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=5, color='green', alpha=0.5)
# ax.plot([land_mean_x], [land_mean_y], [land_mean_z], 'o', markersize=10, color='red', alpha=0.5)
# # plot segmented points
# # ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.75, color='blue', alpha=0.3)
# ax.plot([seg_mean_x], [seg_mean_y], [seg_mean_z], 'o', markersize=10, color='green', alpha=0.2)
#
# for vl, vs in zip(desc_land_eig_vec.T[1:2], desc_seg_eig_vec.T[1:2]):
#     # landmarks' eigenvectors
#     # vl = 4000*vl
#     # al = Arrow3D([land_mean_x, vl[0]], [land_mean_y, vl[1]], [land_mean_z, vl[2]],
#     #              mutation_scale=1, lw=2, arrowstyle="-|>", color="r")
#     # print("Original Eigenvector: \t", vl)
#     # rotating the vector here
#     # vl = x_rotation(vl, 2)    # works for 2nd eigenvector
#     # vl = -z_rotation(vl, 125)    # works for 3rd eigenvector
#     # print(land_mean_x+vl[0], land_mean_y+vl[1], land_mean_z+vl[2])
#     ax.plot([land_mean_x, land_mean_x+vl[0]], [land_mean_y, land_mean_y+vl[1]], [land_mean_z, land_mean_z+vl[2]],
#             color='red', alpha=0.8, lw=2)
#
#     # segmented points' eigenvectors
#     # vs = 4000*vs
#     # aseg = Arrow3D([seg_mean_x, vs[0]], [seg_mean_y, vs[1]], [seg_mean_z, vs[2]],
#     #             mutation_scale=1, lw=2, arrowstyle="-", color="yellow")
#     # print(seg_mean_x+vs[0], seg_mean_y+vs[1], seg_mean_z+vs[2])
#     ax.plot([seg_mean_x, seg_mean_x+vs[0]], [seg_mean_y, seg_mean_y+vs[1]], [seg_mean_z, seg_mean_z+vs[2]],
#             '--', color='green', alpha=0.8, lw=2)
#
# ax.set_xlabel('x_values')
# ax.set_ylabel('y_values')
# ax.set_zlabel('z_values')
# plt.title('Eigenvectors')
# plt.show(); sys.exit()

transformed_desc_land_eig_vec = []
i = 1
for vl, vs in zip(desc_land_eig_vec.T, desc_seg_eig_vec.T):
    # landmarks' eigenvectors
    if i == 1:
        print("Original Eigenvector: \t", vl)
        transformed_desc_land_eig_vec.append(vl)
    elif i==2:
        print("Original Eigenvector: \t", vl)
        vl = x_rotation(vl, 2)
        print("After Rotation: \t", vl)
        transformed_desc_land_eig_vec.append(vl)
    else:
        print("Original Eigenvector: \t", vl)
        vl = -z_rotation(vl, 125)
        print("After Rotation: \t", vl)
        transformed_desc_land_eig_vec.append(vl)
    i += 1

# print("Original landmarks' Eigenvectors: \n", desc_land_eig_vec)
transformed_desc_land_eig_vec = (np.array(transformed_desc_land_eig_vec)).T
# print("New Landmark Eigenvectors: \n", transformed_desc_land_eig_vec)

# ------- TRANSFORMING THE LANDMARK POINTS TO THE ALIGNED SPACE --------
transformed_array = transformed_desc_land_eig_vec.T.dot(array)

# # computing the 3-dimensional mean vector
# tf_land_mean_x = np.mean(transformed_array[0,:])
# tf_land_mean_y = np.mean(transformed_array[1,:])
# tf_land_mean_z = np.mean(transformed_array[2,:])
# tf_land_mean_vec = np.array([[tf_land_mean_x], [tf_land_mean_y], [tf_land_mean_z]])     # 3x1
# print("Centroid Coordinates: \n", tf_land_mean_vec)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot landmark points
# ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=5, color='green', alpha=0.5)
ax.plot([land_mean_x], [land_mean_y], [land_mean_z], 'o', markersize=10, color='red', alpha=0.5)
# plot segmented points
# ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.75, color='blue', alpha=0.3)
ax.plot([seg_mean_x], [seg_mean_y], [seg_mean_z], 'o', markersize=10, color='green', alpha=0.2)

for vl, vs in zip(transformed_desc_land_eig_vec.T[:1], desc_seg_eig_vec.T[:1]):
    # landmarks' eigenvectors
    # print(land_mean_x+vl[0], land_mean_y+vl[1], land_mean_z+vl[2])
    ax.plot([land_mean_x, land_mean_x+vl[0]], [land_mean_y, land_mean_y+vl[1]], [land_mean_z, land_mean_z+vl[2]],
            color='red', alpha=0.8, lw=2)

    # segmented points' eigenvectors
    # print(seg_mean_x+vs[0], seg_mean_y+vs[1], seg_mean_z+vs[2])
    ax.plot([seg_mean_x, seg_mean_x+vs[0]], [seg_mean_y, seg_mean_y+vs[1]], [seg_mean_z, seg_mean_z+vs[2]],
            '--', color='green', alpha=0.8, lw=2)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show(); sys.exit()


# # for 3D SCATTER PLOTS
# data = [
#     go.Scatter3d(x=array[0], y=array[1], z=array[2], mode='markers', marker=dict(size=3, color='royalblue')),
#     go.Scatter3d(x=transformed_array[0], y=transformed_array[1], z=transformed_array[2], mode='markers',
#                  marker=dict(size=5, color='teal')),
#     ]
# fig = go.Figure(data)
# plotly.offline.plot(fig, filename="orig_and_transformed.html", auto_open=True)



# # for 3D SCATTER PLOTS
# data = [
#     go.Scatter3d(x=vol_data[0], y=vol_data[1], z=vol_data[2], mode='markers', marker=dict(size=1, color='lightgreen')),
#     go.Scatter3d(x=[seg_mean_x], y=[seg_mean_y], z=[seg_mean_z], mode='markers', marker=dict(size=5, color='maroon')),
#     go.Scatter3d(x=array[0], y=array[1], z=array[2], mode='markers', marker=dict(size=3, color='royalblue')),
#     go.Scatter3d(x=[land_mean_x], y=[land_mean_y], z=[land_mean_z], mode='markers', marker=dict(size=5, color='teal')),
#     ]
#
# fig = go.Figure(data)
# plotly.offline.plot(fig, filename="combined_plots_1.html", auto_open=True)