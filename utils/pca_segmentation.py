import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import plotly
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
plotly.offline.init_notebook_mode(connected=True)

path = './vol7_pat11_vertT7_coords.npy'
vol_data = np.load(path)
vol_data = vol_data.T
# print(vol_data.shape); sys.exit()   # shape -> 3x11472

# # Plotting the landmarks
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
# ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.5, color='blue', alpha=0.5, label='class1')
# ax.legend(loc='upper right')
# plt.show()
# # sys.exit()

# scaling and centering the data
vol_data = vol_data.T   # the standard scaler method standardizes according to each feature, where each feature is
# the 2nd dimension i.e. (N, features). Therefore, it had to be transposed.
scaler = StandardScaler()
vol_data = (scaler.fit_transform(vol_data)).T

# # Plotting the STANDARDIZED landmarks
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
# ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.5, color='blue', alpha=0.5, label='class1')
# ax.legend(loc='upper right')
# plt.show()
# sys.exit()


# computing the 3-dimensional mean vector
mean_x = np.mean(vol_data[0,:])
mean_y = np.mean(vol_data[1,:])
mean_z = np.mean(vol_data[2,:])
mean_vec = np.array([[mean_x], [mean_y], [mean_z]])     # 3x1
print("Centroid coordinates: \n", mean_vec) #; sys.exit()

# scatter_matrix = np.zeros((3,3))
# for i in range(vol_data.shape[1]):
#     scatter_matrix += (vol_data[:,i].reshape(3,1) - mean_vec).dot((vol_data[:,i].reshape(3,1) - mean_vec).T)
# print('Scatter Matrix:\n', scatter_matrix) #; sys.exit()

# # eigenvalues and eigenvectors of the scatter matrix
# eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
# print("Eigenvalues from scatter matrix: \n", eig_val_sc)
# print("Scatter Matrix Eigenvectors' shape: \n", eig_vec_sc.shape)

# getting the covariance matrix
cov_mat = np.cov([vol_data[0, :], vol_data[1, :], vol_data[2, :]])
print('Covariance Matrix:\n', cov_mat)

# eigenvalues and eigenvectors of the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
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
ax.plot(vol_data[0,:], vol_data[1,:], vol_data[2,:], 'o', markersize=0.5, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=5, color='red', alpha=0.5)
for v in eig_vec_cov.T:
    # v = 4*v
    print(v)
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=1, lw=2, arrowstyle="-|>", color="b")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors - Segmented Points')
plt.show()

# # for 3D Scatter Plots
# data = [
#     go.Scatter3d(x=vol_data[0], y=vol_data[1], z=vol_data[2], mode='markers', marker=dict(size=1, color='lightgreen')),
#     go.Scatter3d(x=[mean_x], y=[mean_y], z=[mean_z], mode='markers', marker=dict(size=5, color='maroon')),
#     ]
#
# fig = go.Figure(data)
# plotly.offline.plot(fig)

# # making a list of eigenvalues, eigenvector tuples
# eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_vec_sc))]
# eig_pairs.sort(key=lambda x: x[0], reverse=True)
# for i in eig_pairs:
#     print(i[0])
#
# # Based on the descending order of the eigenvalues, choose top-k eigenvectors and those are the principal components
# matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
# print('Matrix W:\n', matrix_w)


# # scaling and centering the data
# scaler = StandardScaler()
# vol_data_standardized = scaler.fit_transform(vol_data_1)
#
# # # for checking how many components are needed
# # pca_1 = PCA().fit(vol_data_1)
# # plt.plot(np.cumsum(pca_1.explained_variance_ratio_))    # this shows that the 2 components can capture upto ~99% of the
# # # variance in the data.
# # plt.xlabel('number of components')
# # plt.ylabel('cumulative explained variance')
# # plt.show()
# # sys.exit()
#
# # doing PCA
# pca = PCA(n_components=3)
# reduced_vol_data = pca.fit_transform(vol_data_standardized)
# print(reduced_vol_data.shape)
#
# print(pca.components_)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
#
# def myplot(reduced, coeffs):
#     xs = reduced[:,0]
#     ys = reduced[:,1]
#     n = coeffs.shape[1]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley)
#     for i in range(n):
#         plt.arrow(0, 0, coeffs[i,0], coeffs[i,1], color='r', alpha=0.5)
#         plt.text(coeffs[i,0], coeffs[i,1], "Var"+str(i+1), color='g')
#
# # plt.xlim(-0.75,0.75)
# # plt.ylim(-0.75,0.75)
# plt.xlabel("Principal Component {}".format(1))
# plt.ylabel("Prinicpal Component {}".format(2))
# plt.grid()
#
# myplot(reduced_vol_data[:, 0:2], np.transpose(pca.components_[0:2,:]))
# plt.show()


