import re
import matplotlib.pyplot as plt
import numpy as np

# path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_models/mr2ct_new_unetUpdated/'
# path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_models/mr2ct_unet_batchsize16/'
# path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/saved_models/mr2ct_spectral/'
path = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/utils/'

lines = []
with open(path+'log_stats.txt', 'rt') as file:
    for line in file:
        lines.append(line)

n = len(lines)
G_loss = []   # every 8th element
G_adversarial_loss = [] # every 13th element
G_cycle_loss = [] # every 18th element
G_gradient_loss = []  # every 23rd element
D_loss = []  # every 27th element
epochs = np.arange(1, n+1)

# for idx in range(len(lines)):
#     temp = [str for str in lines[idx].split()]
#     if idx < 100: # 99: # 89:
#         G_loss.append(float(temp[10+1]))
#         G_adversarial_loss.append(float(temp[15+1]))
#         G_cycle_loss.append(float(temp[20+1]))
#         G_identity.append(float(temp[25]))
#         D_loss.append(float(temp[29]))
#     else:
#         G_loss.append(float(temp[8]))
#         G_adversarial_loss.append(float(temp[13]))
#         G_cycle_loss.append(float(temp[18]))
#         G_identity.append(float(temp[23]))
#         D_loss.append(float(temp[27]))

for idx in range(len(lines)):
    temp = [str for str in lines[idx].split()]
    if idx < 164: # 99: # 89:
        G_adversarial_loss.append(float(temp[11]))
        G_cycle_loss.append(float(temp[16]))
        G_gradient_loss.append(float(temp[21]))
        G_loss.append(float(temp[25]))
        D_loss.append(float(temp[29]))
    else:
        print(idx)
        G_adversarial_loss.append(float(temp[9]))
        G_cycle_loss.append(float(temp[14]))
        G_gradient_loss.append(float(temp[19]))
        G_loss.append(float(temp[23]))
        D_loss.append(float(temp[27]))



# print(G_loss)
# print(G_cycle_loss)

plt.plot(epochs, G_adversarial_loss, 'c1-', label='Generator Adversarial Loss', markersize=4, markevery=4)
plt.plot(epochs, G_cycle_loss, 'g+-', label='Cycle Consistency Loss', markersize=4, markevery=4)
plt.plot(epochs, G_gradient_loss, 'rx-', label='Gradient Consistency Loss', markersize=4, markevery=4)
plt.plot(epochs, G_loss, 'bo-', label='Generator Loss', markersize=4, markevery=4)
plt.plot(epochs, D_loss, 'ys-', label='Discriminator Loss', markersize=4, markevery=4)
plt.xlabel('Epochs')
plt.ylabel('Loss Values')
# plt.ylim([0, 3])
plt.legend()
plt.title('Losses Vs. Epochs')
plt.show()

