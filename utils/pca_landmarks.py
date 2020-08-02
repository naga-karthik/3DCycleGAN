import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# this getting the landmarks from .o3 file for each patient.
dir_path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/ANONwithDICOM/PATIENT11/2007-07-25/'
file_name = 'LIS_PATIENT11_edited_thoracic.o3'
# file_name = 'LIS_PATIENT11_edited_thoracic_1.o3'
# file_name = 'test_1.o3'
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
                        x_coord.append(float(data[1]) + 45.27613014)
                        y_coord.append(float(data[3]) - 24.5141186)
                        z_coord.append(float(data[4].rstrip(" \n")) - 343.8056481)

                        # x_coord.append(float(data[1]))
                        # y_coord.append(float(data[3]))
                        # z_coord.append(float(data[4].rstrip(" \n")) )
                        # # z_coord.append(-1.0*float(data[4].rstrip(" \n")))
                    # the z-coordinates are negated because the segmented images are the mirror images of the original
                    # images (that is how the model generated the synthesized images)
                    else:
                        x_coord.append(float(data[1]) + 45.27613014)
                        y_coord.append(float(data[2]) - 24.5141186)
                        z_coord.append(float(data[3].rstrip(" \n")) - 343.8056481)

                        # x_coord.append(float(data[1]))
                        # y_coord.append(float(data[2]))
                        # z_coord.append(float(data[3].rstrip(" \n")) )
                        # # z_coord.append(-1.0*float(data[3].rstrip(" \n")))
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

# print(result)
# sys.exit()
xdata, ydata, zdata = [], [], []
for k, v in landmarks.items():
    xdata.append(v[0])
    ydata.append(v[1])
    zdata.append(v[2])
# # print(xdata)
array = np.array([xdata, ydata, zdata])     # shape = 73x1x3
array = array.squeeze()      # shape = 3x73

# computing the 3-dimensional mean vector
mean_x = np.mean(array[0,:])
mean_y = np.mean(array[1,:])
mean_z = np.mean(array[2,:])
mean_vec = np.array([[mean_x], [mean_y], [mean_z]])     # 3x1
print("Centroid Coordinates: \n", mean_vec)

# Plotting the landmarks
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.legend(loc='upper right')
plt.show()

# --------------- CREATING A ROTATION MATRIX ---------------
theta = np.radians(90)
cos, sin = np.cos(theta), np.sin(theta)
R_yz = np.array(((1, 0, 0), (0, cos, -sin), (0, sin, cos)))
print(R_yz, R_yz.shape)
# multiplying the rotation matrix with the landmark points
array_1 = array.T   # shape = 73x3
rotated_array = (np.dot(array_1, R_yz.T)).T

# computing the 3-dimensional mean vector
mean_x = np.mean(rotated_array[0,:])
mean_y = np.mean(rotated_array[1,:])
mean_z = np.mean(rotated_array[2,:])
mean_vec = np.array([[mean_x], [mean_y], [mean_z]])     # 3x1
print("Centroid Coordinates - 1: \n", mean_vec)

# Plotting the landmarks
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(rotated_array[0,:], rotated_array[1,:], rotated_array[2,:], 'o', markersize=8,
        color='green', alpha=0.5, label='class1')
ax.legend(loc='upper right')
plt.show()
sys.exit()
# -------------------------

# scaling and centering the data
# scaler = StandardScaler()
# landmarks_standardized = scaler.fit_transform(array)

# computing the 3-dimensional mean vector
mean_x = np.mean(array[0,:])
mean_y = np.mean(array[1,:])
mean_z = np.mean(array[2,:])
mean_vec = np.array([[mean_x], [mean_y], [mean_z]])     # 3x1
print("Centroid Coordinates: \n", mean_vec)

scatter_matrix = np.zeros((3,3))
for i in range(array.shape[1]):
    scatter_matrix += (array[:,i].reshape(3,1) - mean_vec).dot((array[:,i].reshape(3,1) - mean_vec).T)
print('Scatter Matrix:\n', scatter_matrix) #; sys.exit()

# eigenvalues and eigenvectors of the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
print("Eigenvalues from scatter matrix: \n", eig_val_sc)
print("Scatter Matrxi Eigenvectors' shape: \n", eig_vec_sc.shape)

# # getting the covariance matrix
# cov_mat = np.cov([array[0, :], array[1, :], array[2, :]])
# print('Covariance Matrix:\n', cov_mat)

# # eigenvalues and eigenvectors of the covariance matrix
# eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
# print("Eigenvalues from covariance matrix: \n", eig_val_cov)
# # the difference between scatter and covariance matrices is the scaling factor -> 1/(N-1)
# # Here, N=73 because there are 73 landmark points

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
ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_sc.T:
    v = 4000*v
    print(v)
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=1, lw=2, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors - Landmark Points')
plt.show()

# # making a list of eigenvalues, eigenvector tuples
# eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_vec_sc))]
# eig_pairs.sort(key=lambda x: x[0], reverse=True)
# for i in eig_pairs:
#     print(i[0])
#
# # Based on the descending order of the eigenvalues, choose top-k eigenvectors and those are the principal components
# matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
# print('Matrix W:\n', matrix_w)

# # scaling and center -> standardizing the data first
# mu = np.mean(array)
# sigma = np.std(array)
# landmarks_standardized = (array - mu)/sigma
# # print(landmarks_standardized[0])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(landmarks_standardized[0], landmarks_standardized[1], landmarks_standardized[2])
# plt.show()

# # for plotting a boxplot of mean MI values for axial, sagittal and coronal slices
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
#
# values = [ ['Axial', 0.4356], ['Axial', 0.4763], ['Axial', 0.3984], ['Axial', 0.5431], ['Axial', 0.3643],
#            ['Axial', 0.4540], ['Axial', 0.4306], ['Axial', 0.4102], ['Axial', 0.4717], ['Axial', 0.4342],
#            ['Axial', 0.4349], ['Axial', 0.3927], ['Axial', 0.4055], ['Axial', 0.4742], ['Axial', 0.3842],
#
#            ['Sagittal', 0.2692], ['Sagittal', 0.3194], ['Sagittal', 0.1832], ['Sagittal', 0.3821],
#            ['Sagittal', 0.1962], ['Sagittal', 0.2656], ['Sagittal', 0.2486], ['Sagittal', 0.2160],
#            ['Sagittal', 0.3059], ['Sagittal', 0.2556], ['Sagittal', 0.2658], ['Sagittal', 0.2145],
#            ['Sagittal', 0.2242], ['Sagittal', 0.3268], ['Sagittal', 0.2218],
#
#            ['Coronal', 0.2370], ['Coronal', 0.3176], ['Coronal', 0.2873], ['Coronal', 0.3795], ['Coronal', 0.2230],
#            ['Coronal', 0.2185], ['Coronal', 0.2670], ['Coronal', 0.2684], ['Coronal', 0.2607], ['Coronal', 0.2871],
#            ['Coronal', 0.2999], ['Coronal', 0.2351], ['Coronal', 0.2930], ['Coronal', 0.3078], ['Coronal', 0.2518]
#           ]
# df = pd.DataFrame(data=values, columns=['Slice Type', 'Mean MI Score'])
# print(df.head())
#
# sns.set_style(style='darkgrid')
# sns.boxplot(x="Slice Type", y="Mean MI Score", data=df)
# plt.show()


