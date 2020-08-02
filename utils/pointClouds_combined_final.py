import sys
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
plotly.offline.init_notebook_mode(connected=True)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# ----------- Landmarks points ----------------
array = np.load('original_vertT7_coords.npy')
# print(array.shape); sys.exit()

# Scaling the values between mean=0 and and sigma=1.
# the standard scaler method standardizes according to each feature, where each feature is
# the 2nd dimension i.e. (N, features). Therefore, it has to be transposed.
scaler = StandardScaler()
array = (scaler.fit_transform(array.T)).T
# sys.exit()

# computing the 3-dimensional mean vector after translation
land_mean_x = np.mean(array[0,:])
land_mean_y = np.mean(array[1,:])
land_mean_z = np.mean(array[2,:])
land_mean_vec = np.array([[land_mean_x], [land_mean_y], [land_mean_z]])     # 3x1
# print("Landmark Mean Coordinates: \n", land_mean_vec)

# getting the covariance matrix
land_cov_mat = np.cov([array[0, :], array[1, :], array[2, :]])
print('Landmarks Covariance Matrix:\n', land_cov_mat)
# eigenvalues and eigenvectors of the covariance matrix
land_eig_val_cov, land_eig_vec_cov = np.linalg.eig(land_cov_mat)

# arranging the eigenvalues and eigenvectors in the descending order.
# making a list of eigenvalues, eigenvector tuples
land_eig_pairs = [(np.abs(land_eig_val_cov[i]), land_eig_vec_cov[:,i]) for i in range(len(land_eig_vec_cov))]
land_eig_pairs.sort(key=lambda x: x[0], reverse=True)
desc_land_eig_vec = [np.hstack((land_eig_pairs[i][1].reshape(3,1))) for i in range(3)]
desc_land_eig_vec = (np.array(desc_land_eig_vec)).T
print("Landmark Eigenvectors Descending: \n", desc_land_eig_vec)


# ---------------------- Segmented Points ---------------------------
path = './vol7_pat11_vertT7_coords.npy'
vol_data = np.load(path)
vol_data = vol_data.T

# Scaling the values between 0 and 1.
scaler = StandardScaler()
vol_data = (scaler.fit_transform(vol_data.T)).T
# sys.exit()

# computing the 3-dimensional mean vector
seg_mean_x = np.mean(vol_data[0,:])
seg_mean_y = np.mean(vol_data[1,:])
seg_mean_z = np.mean(vol_data[2,:])
seg_mean_vec = np.array([[seg_mean_x], [seg_mean_y], [seg_mean_z]])     # 3x1
# print("Segmented Mean Coordinates: \n", seg_mean_vec) #; sys.exit()

# getting the covariance matrix
seg_cov_mat = np.cov([vol_data[0, :], vol_data[1, :], vol_data[2, :]])
print('Segmentation Covariance Matrix:\n', seg_cov_mat)
# eigenvalues and eigenvectors of the covariance matrix
seg_eig_val_cov, seg_eig_vec_cov = np.linalg.eig(seg_cov_mat)

# arranging the eigenvalues and eigenvectors in the descending order.
# making a list of eigenvalues, eigenvector tuples
seg_eig_pairs = [(np.abs(seg_eig_val_cov[i]), seg_eig_vec_cov[:,i]) for i in range(len(seg_eig_vec_cov))]
seg_eig_pairs.sort(key=lambda x: x[0], reverse=True)
# print(seg_eig_pairs)
desc_seg_eig_vec = [np.hstack((seg_eig_pairs[i][1].reshape(3,1))) for i in range(3)]
desc_seg_eig_vec = (np.array(desc_seg_eig_vec)).T
print("Segmentation Eigenvectors Descending: \n", desc_seg_eig_vec)
# print("Largest Eigenvector: \n", desc_seg_eig_vec[:1])
# print("Second Eigenvector: \n", desc_seg_eig_vec[1:2])
# print("Last Eigenvector: \n", desc_seg_eig_vec[2:3])
# sys.exit()


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
# ax.plot(array[0,:], array[1,:], array[2,:], 'o', markersize=5, color='green', alpha=0.5)
# ax.plot([land_mean_x], [land_mean_y], [land_mean_z], 'o', markersize=10, color='red', alpha=0.5)
# # plot segmented points
# ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.75, color='blue', alpha=0.3)
# ax.plot([seg_mean_x], [seg_mean_y], [seg_mean_z], 'o', markersize=10, color='green', alpha=0.2)
# # plt.show()
# #
# for vl, vs in zip(desc_land_eig_vec.T[2:3], desc_seg_eig_vec.T[2:3]):
#     # landmarks' eigenvectors
#     # vl = -z_rotation(vl, 130)   # works for the 1st eigenvector
#     # vl = z_rotation(vl, -55)   # works for the 2nd eigenvector
#     # vl = y_rotation(vl, -110)   # works for the 3rd eigenvector
#     # vl = 3*vl
#     al = Arrow3D([land_mean_x, vl[0]], [land_mean_y, vl[1]], [land_mean_z, vl[2]],
#                  mutation_scale=1, lw=2, arrowstyle="-|>", color="r")
#     # print("Original Eigenvector: \t", vl)
#     # rotating the vector here
#     # vl = x_rotation(vl, 2)    # works for 2nd eigenvector
#     # vl = -z_rotation(vl, 125)    # works for 3rd eigenvector
#     # ax.plot([land_mean_x, land_mean_x+vl[0]], [land_mean_y, land_mean_y+vl[1]], [land_mean_z, land_mean_z+vl[2]],
#     #         color='red', alpha=0.8, lw=2)
#
#     # segmented points' eigenvectors
#     # vs = 3*vs
#     aseg = Arrow3D([seg_mean_x, vs[0]], [seg_mean_y, vs[1]], [seg_mean_z, vs[2]],
#                 mutation_scale=1, lw=2, arrowstyle="-", color="yellow")
#     # ax.plot([seg_mean_x, seg_mean_x+vs[0]], [seg_mean_y, seg_mean_y+vs[1]], [seg_mean_z, seg_mean_z+vs[2]],
#     #         '--', color='green', alpha=0.8, lw=2)
#     ax.add_artist(al)
#     ax.add_artist(aseg)
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
        # print("Original Eigenvector: \t", vl)
        vl = -z_rotation(vl, 130)
        transformed_desc_land_eig_vec.append(vl)
    elif i==2:
        # print("Original Eigenvector: \t", vl)
        vl = z_rotation(vl, -55)
        # print("After Rotation: \t", vl)
        transformed_desc_land_eig_vec.append(vl)
    else:
        # print("Original Eigenvector: \t", vl)
        vl = y_rotation(vl, -110)
        # print("After Rotation: \t", vl)
        transformed_desc_land_eig_vec.append(vl)
    i += 1

# print("Original landmarks' Eigenvectors: \n", desc_land_eig_vec)
transformed_desc_land_eig_vec = (np.array(transformed_desc_land_eig_vec)).T
# print("New Landmark Eigenvectors: \n", transformed_desc_land_eig_vec)

# ------- TRANSFORMING THE LANDMARK POINTS TO THE ALIGNED SPACE --------
transformed_array = transformed_desc_land_eig_vec.T.dot(array)
np.save('./transformed_vertT7_coords', transformed_array)

# computing the 3-dimensional mean vector
tf_land_mean_x = np.mean(transformed_array[0,:])
tf_land_mean_y = np.mean(transformed_array[1,:])
tf_land_mean_z = np.mean(transformed_array[2,:])
tf_land_mean_vec = np.array([[tf_land_mean_x], [tf_land_mean_y], [tf_land_mean_z]])     # 3x1
print("Transformed Centroid Coordinates: \n", tf_land_mean_vec)


# ---- PLOTTING THE TRANSFORMED EIGENVECTORS ALONG WITH THE SEGMENTED POINTS' EIGENVECTORS ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot landmark points
ax.plot(transformed_array[0,:], transformed_array[1,:], transformed_array[2,:],
        'o', markersize=5, color='green', alpha=0.5)
ax.plot([tf_land_mean_x], [tf_land_mean_y], [tf_land_mean_z], 'o', markersize=10, color='red', alpha=0.5)
# plot segmented points
ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.75, color='blue', alpha=0.3)
ax.plot([seg_mean_x], [seg_mean_y], [seg_mean_z], 'o', markersize=10, color='green', alpha=0.2)

for vl, vs in zip(transformed_desc_land_eig_vec.T[:], desc_seg_eig_vec.T[:]):
    # landmarks' eigenvectors
    # vl = 3*vl
    al = Arrow3D([tf_land_mean_x, vl[0]], [tf_land_mean_y, vl[1]], [tf_land_mean_z, vl[2]],
                 mutation_scale=1, lw=2, arrowstyle="-|>", color="r")
    # segmented points' eigenvectors
    # vs = 4*vs
    aseg = Arrow3D([seg_mean_x, vs[0]], [seg_mean_y, vs[1]], [seg_mean_z, vs[2]],
                   mutation_scale=1, lw=2, arrowstyle="-", color="yellow")
    ax.add_artist(al)
    ax.add_artist(aseg)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show(); sys.exit()


# # for 3D SCATTER PLOTS
# data = [
#     # go.Scatter3d(x=array[0], y=array[1], z=array[2], mode='markers', marker=dict(size=3, color='royalblue')),
#     go.Scatter3d(x=transformed_array[0], y=transformed_array[1], z=transformed_array[2], mode='markers',
#                  marker=dict(size=5, color='teal')),
#     ]
# fig = go.Figure(data)
# plotly.offline.plot(fig, filename="orig_and_transformed.html", auto_open=True)

# # for 3D SCATTER PLOTS
# data = [
#     go.Scatter3d(x=vol_data[0], y=vol_data[1], z=vol_data[2], mode='markers', marker=dict(size=1, color='lightgreen')),
#     go.Scatter3d(x=[seg_mean_x], y=[seg_mean_y], z=[seg_mean_z], mode='markers', marker=dict(size=5, color='maroon')),
#     go.Scatter3d(x=transformed_array[0], y=transformed_array[1], z=transformed_array[2],
#                  mode='markers', marker=dict(size=3, color='royalblue')),
#     go.Scatter3d(x=[tf_land_mean_x], y=[tf_land_mean_y], z=[tf_land_mean_z],
#                  mode='markers', marker=dict(size=5, color='teal')),
#     ]
# fig = go.Figure(data)
# plotly.offline.plot(fig, filename="combined_plots_tfd.html", auto_open=True)