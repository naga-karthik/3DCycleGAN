import glob, os, sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', one_side=False):
        self.transform = transforms.Compose(transform)
        # self.unaligned = unaligned
        self.mode = mode
        self.one_side = one_side
        self.files_A = sorted(glob.glob(os.path.join(root_dir, '%s/A' % self.mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, '%s/B' % self.mode) + '/*.*'))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # NOTE: the data is transformed into a tensor in [z, x, y] format. Because PyTorch puts the z-axis first.
        # Therefore, when shape is printed, the format is [batch_size x Z x X x Y] or, [batch_size x 48 x 256 x 128]
        # item_A = self.transform(np.load(self.files_A[idx % len(self.files_A)]))
        # item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B)-1)]))

        if (self.mode == 'test' or self.mode == "test_1") and self.one_side:
            item_A = self.transform(
                torch.FloatTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2, 0, 1)))
            return {'A': item_A}
        else:
            item_A = self.transform(
                torch.FloatTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2, 0, 1)))
            item_B = self.transform(
                torch.FloatTensor(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])).permute((2, 0, 1)))

            # unsqueezing to add the channel dimension
            item_A = item_A.unsqueeze(0)
            item_B = item_B.unsqueeze(0)

            return {'A': item_A, 'B': item_B}


# Testing the code here
if __name__ == '__main__':
    dataroot = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/'
    transforms_ = [transforms.Normalize((0.0,), (1.0,))]
    # since the image is already in [-1, 1] range, we normalize it by mu=0 and sigma=1. transforms.Normalize just
    # performs this operation: img = (img-mu)/sigma

    # there is doubt whether we should normalize an image that is already in [-1, 1] range. CHECK THIS
    data = ImageDataset(root_dir=dataroot, transform=transforms_)
    data_loader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    temp = next(iter(data_loader))
    # print(temp['A'].min(), temp['A'].max())
    print(temp['B'].shape)
    # output shape when patches=True -> [batch_size, num_patches, patch_size, patch_size, patch_size]
    # output shape when patches=False -> [batch_size, z, x, y]
    # img = temp['B']
    # print(img.shape)
    # plt.figure(); plt.imshow(img[1, 4, :, :, 20], cmap='gray'); plt.show()
    # unf_img = temp.fold(2, 32, 32).fold(3, 32, 32).fold(4, 32, 32)
    # print(unf_img.shape)
    # # plt.figure(); plt.imshow(unf_img[1, :, :, 20], cmap='gray'); plt.show()



