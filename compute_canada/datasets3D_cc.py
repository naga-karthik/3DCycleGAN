import glob, os, sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', patches=False):
        self.transform = transforms.Compose(transform)
        # self.unaligned = unaligned
        self.patches = patches

        # This is for returning NON-OVERLAPPING patches. kd = kernel-size for depth dim, kh = kernel height and so on.
        self.kd, self.kh, self.kw = 48, 64, 64   # if patches=True, then the dataloader returns patches of 32 x 32 x 32
        self.sd, self.sh, self.sw = 48, 64, 64

        self.files_A = sorted(glob.glob(os.path.join(root_dir, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, '%s/B' % mode) + '/*.*'))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # NOTE: the data is transformed into a tensor in [z, x, y] format. Because PyTorch puts the z-axis first.
        # Therefore, when shape is printed, the format is [batch_size x Z x X x Y] or, [batch_size x 48 x 256 x 128]
        # item_A = self.transform(np.load(self.files_A[idx % len(self.files_A)]))
        # item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B)-1)]))

        item_A = self.transform(torch.DoubleTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2,0,1)))
        item_B = self.transform(torch.DoubleTensor(np.load(self.files_B[random.randint(0, len(self.files_B)-1)])).permute((2,0,1)))

        # unsqueezing to add the channel dimension
        item_A = item_A.unsqueeze(0)
        item_B = item_B.unsqueeze(0)

        # if self.unaligned:
        #     item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B)-1)]))
        # else:
        #     item_B = self.transform(np.load(self.files_B[idx % len(self.files_B)]))

        if not self.patches:
            return {'A': item_A,
                    'B': item_B}
        else:
            # unfold is command to create non-overlapping patches of the input data. The resulting number of patches
            # is given by this formula: (Z//patch_size * X//patch_size * Y//patch_size). For patch_size=32, that number
            # comes out to be 32 (because, 1*8*4=32)
            # print(item_A.shape)
            # patch_item_A = item_A.\
            #     unfold(1, self.patch_size, self.patch_size).\
            #     unfold(2, self.patch_size, self.patch_size).\
            #     unfold(3, self.patch_size, self.patch_size)
            # # patch_item_A = patch_item_A.contiguous().view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
            # patch_item_A = patch_item_A.contiguous().view(-1, self.patch_size, self.patch_size, self.patch_size)
            #
            # patch_item_B = item_B.\
            #     unfold(1, self.patch_size, self.patch_size).\
            #     unfold(2, self.patch_size, self.patch_size).\
            #     unfold(3, self.patch_size, self.patch_size)
            # # patch_item_B = patch_item_B.contiguous().view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
            # patch_item_B = patch_item_B.contiguous().view(-1, self.patch_size, self.patch_size, self.patch_size)

            patch_item_A = item_A.unfold(1, self.kd, self.sd).unfold(2, self.kh, self.sh).unfold(3, self.kw, self.sw)
            # print(patch_item_A.size())
            patch_A_shape = patch_item_A.size()
            patch_item_A = patch_item_A.contiguous().view(patch_item_A.size(0), -1, self.kd, self.kh, self.kw)
            patch_item_A = patch_item_A.squeeze()
            # print(patch_item_A.shape)

            patch_item_B = item_B.unfold(1, self.kd, self.sd).unfold(2, self.kh, self.sh).unfold(3, self.kw, self.sw)
            patch_B_shape = patch_item_B.size()
            patch_item_B = patch_item_B.contiguous().view(patch_item_B.size(0), -1, self.kd, self.kh, self.kw)
            patch_item_B = patch_item_B.squeeze()
            # print(patch_item_B.shape)

            # # Code for reshaping the patches back to the original image
            # original_A = patch_item_A.view(patch_A_shape)
            # outd = patch_A_shape[1] * patch_A_shape[4]
            # outh = patch_A_shape[2] * patch_A_shape[5]
            # outw = patch_A_shape[3] * patch_A_shape[6]
            # original_A = original_A.permute(0, 1,4, 2,5, 3,6).contiguous()
            # # print(original_A.shape)
            # original_A = original_A.view(1, outd, outh, outw)
            # # print((original_A == item_A[:, :outd, :outh, :outw]).all())
            #
            # original_B = patch_item_B.view(patch_B_shape)
            # outd = patch_B_shape[1] * patch_B_shape[4]
            # outh = patch_B_shape[2] * patch_B_shape[5]
            # outw = patch_B_shape[3] * patch_B_shape[6]
            # original_B = original_B.permute(0, 1,4, 2,5, 3,6).contiguous()
            # original_B = original_B.view(1, outd, outh, outw)
            # # print((original_B == item_B[:, :outd, :outh, :outw]).all())

            return {'A': patch_item_A,
                    'B': patch_item_B}


# Testing the code here
if __name__ == '__main__':
    dataroot = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct_new/'
    transforms_ = [transforms.Normalize((0.0,), (1.0,))]
    # since the image is already in [-1, 1] range, we normalize it by mu=0 and sigma=1. transforms.Normalize just
    # performs this operation: img = (img-mu)/sigma

    # there is doubt whether we should normalize an image that is already in [-1, 1] range. CHECK THIS
    data = ImageDataset(root_dir=dataroot, transform=transforms_, patches=True)
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



