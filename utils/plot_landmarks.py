import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
plotly.offline.init_notebook_mode(connected=True)
import sys

# this file is for plotting the landmarks from .o3 file for each patient.

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
        if name != 'Vertebre_T12':      # each vertebra has 73 points, except for T12 which has 69 points
            for j in range(3, 76):
                if not lines[i+j].startswith("#"):  # ignore lines that start with '#'
                    data = lines[i+j].split("   ")
                    # print(data, len(data))
                    # sys.exit()
                    if len(data) == 5:
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[3]))
                        z_coord.append(-1.0*float(data[4].rstrip(" \n")))
                    # the z-coordinates are negated because the segmented images are the mirror images of the original
                    # images (that is how the model generated the synthesized images)
                    else:
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[2]))
                        z_coord.append(-1.0*float(data[3].rstrip(" \n")))
                else:
                    continue
            landmarks.update({name: (x_coord, y_coord, z_coord)})

        else:
            for j in range(3, 72):
                if not lines[i+j].startswith("#"):
                    data = lines[i+j].split("   ")
                    # print(data, len(data))
                    # sys.exit()
                    if len(data) == 5:
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[3]))
                        z_coord.append(-1.0*float(data[4].rstrip(" \n")))
                    else:
                        x_coord.append(float(data[1]))
                        y_coord.append(float(data[2]))
                        z_coord.append(-1.0*float(data[3].rstrip(" \n")))
                else:
                    continue
            # landmarks.update({
            #     name + "_x": x_coord,
            #     name + "_y": y_coord,
            #     name + "_z": z_coord,
            # })
            landmarks.update({ name: (x_coord, y_coord, z_coord)})

# print(landmarks)
# sys.exit()

# fig = plt.figure()
# print(landmarks['Vertebre_T1'])
# xdata = landmarks['Vertebre_T1'][0]
# ydata = landmarks['Vertebre_T1'][1]
# zdata = landmarks['Vertebre_T1'][2]
xdata, ydata, zdata = [], [], []
for k, v in landmarks.items():
    xdata.append(v[0])
    ydata.append(v[1])
    zdata.append(v[2])

# # experiments
# y_curr = ydata[0]
# print(y_curr)
# y_curr1 = []
# print(len(y_curr))
# for i in range(len(y_curr)):
#     if y_curr[i] < 20.0 and y_curr[i] > -35.0:
#     # why this is not working is the corresponding elements in x and z should also be removed.
#         y_curr1.append(y_curr[i])
#
# print(len(y_curr1))
# print(y_curr1)
# # sys.exit()

# # for 3D Scatter Plots
# data = [
#     go.Scatter3d(x=xdata[0], y=ydata[0], z=zdata[0], mode='markers', marker=dict(size=5, color='lightgreen')),
#     go.Scatter3d(x=xdata[1], y=ydata[1], z=zdata[1], mode='markers', marker=dict(size=5, color='lightpink')),
#     ]
# for 3D Scatter Plots
data = [
    go.Scatter3d(x=xdata[0], y=ydata[0], z=zdata[0], mode='markers', marker=dict(size=4, color='lightgreen')),
    go.Scatter3d(x=xdata[1], y=ydata[1], z=zdata[1], mode='markers', marker=dict(size=4, color='lightpink')),
    go.Scatter3d(x=xdata[2], y=ydata[2], z=zdata[2], mode='markers', marker=dict(size=4, color='lightsalmon')),
    go.Scatter3d(x=xdata[3], y=ydata[3], z=zdata[3], mode='markers', marker=dict(size=4, color='lightskyblue')),
    go.Scatter3d(x=xdata[4], y=ydata[4], z=zdata[4], mode='markers', marker=dict(size=4, color='lightslategray')),
    go.Scatter3d(x=xdata[5], y=ydata[5], z=zdata[5], mode='markers', marker=dict(size=4, color='yellow')),
    go.Scatter3d(x=xdata[6], y=ydata[6], z=zdata[6], mode='markers', marker=dict(size=4, color='teal')),
    go.Scatter3d(x=xdata[7], y=ydata[7], z=zdata[7], mode='markers', marker=dict(size=4, color='maroon')),
    go.Scatter3d(x=xdata[8], y=ydata[8], z=zdata[8], mode='markers', marker=dict(size=4, color='gold')),
    go.Scatter3d(x=xdata[9], y=ydata[9], z=zdata[9], mode='markers', marker=dict(size=4, color='royalblue')),
    go.Scatter3d(x=xdata[10], y=ydata[10], z=zdata[10], mode='markers', marker=dict(size=4, color='turquoise')),
    go.Scatter3d(x=xdata[11], y=ydata[11], z=zdata[11], mode='markers', marker=dict(size=4, color='seagreen')),
    ]

# data = [
#     go.Mesh3d(x=xdata[0], y=y_curr1, z=zdata[0], alphahull=3, color='lightgreen', opacity=0.75),
#     go.Mesh3d(x=xdata[1], y=ydata[1], z=zdata[1], alphahull=7, color='lightpink', opacity=0.75),
#     ]
# # for 3D Mesh plots
# print(len(xdata))   # all 12 thoracic vertebrae
# data = [
#     go.Mesh3d(x=xdata[0], y=ydata[0], z=zdata[0], alphahull=7, color='lightgreen', opacity=0.75),
#     go.Mesh3d(x=xdata[1], y=ydata[1], z=zdata[1], alphahull=7, color='lightpink', opacity=0.75),
#     go.Mesh3d(x=xdata[2], y=ydata[2], z=zdata[2], alphahull=7, color='lightsalmon', opacity=0.75),
#     go.Mesh3d(x=xdata[3], y=ydata[3], z=zdata[3], alphahull=7, color='lightskyblue', opacity=0.75),
#     go.Mesh3d(x=xdata[4], y=ydata[4], z=zdata[4], alphahull=7, color='lightslategray', opacity=0.75),
#     go.Mesh3d(x=xdata[5], y=ydata[5], z=zdata[5], alphahull=7, color='yellow', opacity=0.75),
#     go.Mesh3d(x=xdata[6], y=ydata[6], z=zdata[6], alphahull=7, color='teal', opacity=0.75),
#     go.Mesh3d(x=xdata[7], y=ydata[7], z=zdata[7], alphahull=7, color='lightcyan', opacity=0.75),
#     go.Mesh3d(x=xdata[8], y=ydata[8], z=zdata[8], alphahull=7, color='gold', opacity=0.75),
#     go.Mesh3d(x=xdata[9], y=ydata[9], z=zdata[9], alphahull=7, color='royalblue', opacity=0.75),
#     go.Mesh3d(x=xdata[10], y=ydata[10], z=zdata[10], alphahull=7, color='turquoise', opacity=0.75),
#     go.Mesh3d(x=xdata[11], y=ydata[11], z=zdata[11], alphahull=7, color='seagreen', opacity=0.75),
#     ]

fig = go.Figure(data)
plotly.offline.plot(fig)
# fig.show()
# ax.plot_wireframe(xdata, ydata, zdata, color='black')
# ax.set_title('wireframe')
# ax.plot3D(xdata, ydata, zdata)
# plt.show()


