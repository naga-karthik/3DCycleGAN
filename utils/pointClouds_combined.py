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

# THIS METHODS ROUGHLY ALIGNS THE POINT CLOUDS WITHOUT PRINICIPAL AXES. IT FIRST ROTATES THE LANDMARK POINTS AND
# THEN TRANSLATES IT TO MAKE ROUGHLY ALIGN WITH THE SEGMENTED POINT CLOUD.

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
                        # those numbers are translation factors for each axis. This is to speicfically align the mean/
                        # centroid of the segmented and the landmark points.
                        # x_coord.append(float(data[1]) + 45.27613014)
                        # y_coord.append(float(data[3]) - 24.5141186)
                        # z_coord.append(float(data[4].rstrip(" \n")) - 343.8056481)

                        # # new translation coordinates after rotation
                        # x_coord.append(float(data[1]) + 19.4886689)
                        # y_coord.append(float(data[3]) - 114.15875211)
                        # z_coord.append(float(data[4].rstrip(" \n")) - 343.8056481)

                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[3]))
                        z_coord.append(float(data[4].rstrip(" \n")))


                    # the z-coordinates are negated because the segmented images are the mirror images of the original
                    # images (that is how the model generated the synthesized images)
                    else:
                        # x_coord.append(float(data[1]) + 45.27613014)
                        # y_coord.append(float(data[2]) - 24.5141186)
                        # z_coord.append(float(data[3].rstrip(" \n")) - 343.8056481)

                        # # new translation coordinates after rotation
                        # x_coord.append(float(data[1]) + 19.4886689)
                        # y_coord.append(float(data[2]) - 114.15875211)
                        # z_coord.append(float(data[3].rstrip(" \n")) - 343.8056481)

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
# print(array.shape)

# --------------- CREATING A ROTATION MATRIX ---------------
theta = np.radians(-60)
cos, sin = np.cos(theta), np.sin(theta)
R_xy = np.array(((cos, -sin, 0), (sin, cos, 0), (0, 0, 1)))   # about Z-axis
# R_yz = np.array(((1, 0, 0), (0, cos, -sin), (0, sin, cos)))     # about X-axis
# R_xz = np.array(((cos, 0, sin), (0, 1, 0), (-sin, 0, cos)))   # about Y-axis
# print(R_xy, R_xy.shape)

# multiplying the rotation matrix with the landmark points
array = np.dot(R_xy, array)

# computing the 3-dimensional mean vector JUST after rotation and use that mean to translate
mean_rot_x = np.mean(array[0,:])
mean_rot_y = np.mean(array[1,:])
mean_rot_z = np.mean(array[2,:])
mean_vec = np.array([[mean_rot_x], [mean_rot_y], [mean_rot_z]])     # 3x1
# print("Centroid Rotated Coordinates: \n", mean_vec)
# -------------------------------------------------------

# ---------------------- Segmented Points ---------------------------
path = './vol7_pat11_vertT7_coords.npy'
vol_data = np.load(path)
vol_data = vol_data.T

# computing the 3-dimensional mean vector
seg_mean_x = np.mean(vol_data[0,:])
seg_mean_y = np.mean(vol_data[1,:])
seg_mean_z = np.mean(vol_data[2,:])
seg_mean_vec = np.array([[seg_mean_x], [seg_mean_y], [seg_mean_z]])     # 3x1
# print("Mean per dimension: \n", seg_mean_vec) #; sys.exit()

# ------- TRANSLATION ----------
trans_x = seg_mean_x - mean_rot_x
trans_y = seg_mean_y - mean_rot_y
trans_z = seg_mean_z - mean_rot_z
trans_vec = np.array([[trans_x], [trans_y], [trans_z]])
array = array + trans_vec

land_mean_x = np.mean(array[0,:])
land_mean_y = np.mean(array[1,:])
land_mean_z = np.mean(array[2,:])
land_mean_vec = np.array([[land_mean_x], [land_mean_y], [land_mean_z]])     # 3x1
# print("Centroid Rotated+Translated Coordinates: \n", land_mean_vec); sys.exit()

# ------ PRINCIPAL COMPONENTS FOR LANDMARK POINTS ------------
scatter_matrix_landmarks = np.zeros((3,3))
for i in range(array.shape[1]):
    scatter_matrix_landmarks += (array[:,i].reshape(3,1) - land_mean_vec).dot((array[:,i].reshape(3,1) - land_mean_vec).T)
print('Landmarks Scatter Matrix:\n', scatter_matrix_landmarks) #; sys.exit()

# eigenvalues and eigenvectors of the scatter matrix
land_eig_val_sc, land_eig_vec_sc = np.linalg.eig(scatter_matrix_landmarks)
print("Eigenvalues from scatter matrix: \n", land_eig_val_sc)
# print("Scatter Matrix Eigenvectors' shape: \n", land_eig_vec_sc.shape)

# ------ PRINCIPAL COMPONENTS FOR SEGMENTED POINTS ------------
scatter_matrix_segment = np.zeros((3,3))
for i in range(vol_data.shape[1]):
    scatter_matrix_segment += (vol_data[:,i].reshape(3,1) - seg_mean_vec).dot((vol_data[:,i].reshape(3,1) - seg_mean_vec).T)
print('Segmentation Scatter Matrix:\n', scatter_matrix_segment) #; sys.exit()

# eigenvalues and eigenvectors of the scatter matrix
seg_eig_val_sc, seg_eig_vec_sc = np.linalg.eig(scatter_matrix_segment)
print("Eigenvalues from scatter matrix: \n", seg_eig_val_sc)
# print("Scatter Matrix Eigenvectors' shape: \n", seg_eig_vec_sc.shape)


# Plotting the eigenvectors centered at the sample mean
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot landmark points
ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=5, color='green', alpha=0.5)
ax.plot([land_mean_x], [land_mean_y], [land_mean_z], 'o', markersize=10, color='red', alpha=0.5)
# plot segmented points
ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.75, color='blue', alpha=0.3)
ax.plot([seg_mean_x], [seg_mean_y], [seg_mean_z], 'o', markersize=10, color='yellow', alpha=0.2)

for vl, vs in zip(land_eig_vec_sc.T[:2], seg_eig_vec_sc.T[:2]):
    vl = 4000*vl
    al = Arrow3D([land_mean_x, vl[0]], [land_mean_y, vl[1]], [land_mean_z, vl[2]],
                mutation_scale=1, lw=2, arrowstyle="-|>", color="r")
    vs = 4000*vs
    aseg = Arrow3D([seg_mean_x, vs[0]], [seg_mean_y, vs[1]], [seg_mean_z, vs[2]],
                mutation_scale=1, lw=2, arrowstyle="-|>", color="yellow")
    ax.add_artist(al)
    ax.add_artist(aseg)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show()

# # for 3D SCATTER PLOTS
# data = [
#     go.Scatter3d(x=vol_data[0], y=vol_data[1], z=vol_data[2], mode='markers', marker=dict(size=1, color='lightgreen')),
#     go.Scatter3d(x=[seg_mean_x], y=[seg_mean_y], z=[seg_mean_z], mode='markers', marker=dict(size=5, color='maroon')),
#     go.Scatter3d(x=array[0], y=array[1], z=array[2], mode='markers', marker=dict(size=3, color='royalblue')),
#     go.Scatter3d(x=[land_mean_x], y=[land_mean_y], z=[land_mean_z], mode='markers', marker=dict(size=5, color='teal')),
#     ]
#
# fig = go.Figure(data)
# plotly.offline.plot(fig, filename="combined_plots.html", auto_open=True)