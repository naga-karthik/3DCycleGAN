import glob, os, sys
import random
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', patches=False, one_side=False):
        self.transform = transforms.Compose(transform)
        self.mode = mode
        self.one_side = one_side
        self.patches = patches
        self.patch_size = 32    # if patches is True, then the dataloader returns patches of 32 x 32 x 32 instead of
                                # the whole volume

        self.files_A = sorted(glob.glob(os.path.join(root_dir, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, '%s/B' % mode) + '/*.*'))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # NOTE: the data is transformed into a tensor in [z, x, y] format. Because PyTorch puts the z-axis first.
        # Therefore, when shape is printed, the format is [batch_size x Z x X x Y] or, [batch_size x 48 x 256 x 128]
        # item_A = self.transform(np.load(self.files_A[idx % len(self.files_A)]))
        # item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B)-1)]))

        if self.mode == 'test' and self.one_side:
            item_A = self.transform(torch.DoubleTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2, 0, 1)))
            return {'A': item_A}

        else:
            item_A = self.transform(torch.DoubleTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2,0,1)))
            item_B = self.transform(torch.DoubleTensor(np.load(self.files_B[random.randint(0, len(self.files_B)-1)])).permute((2,0,1)))

            # unsqueezing to add the channel dimension
            item_A = item_A.unsqueeze(0)
            item_B = item_B.unsqueeze(0)

            if not self.patches:
                return {'A': item_A,
                        'B': item_B}
            else:
                # unfold is command to create non-overlapping patches of the input data. The resulting number of patches
                # is given by this formula: (Z//patch_size * X//patch_size * Y//patch_size). For patch_size=32, that number
                # comes out to be 32 (because, 1*8*4=32)
                # print(item_A.shape)
                patch_item_A = item_A.\
                    unfold(3, self.patch_size, self.patch_size).\
                    unfold(2, self.patch_size, self.patch_size).\
                    unfold(1, self.patch_size, self.patch_size)
                patch_item_A = patch_item_A.contiguous().view(-1, 1, self.patch_size, self.patch_size, self.patch_size)

                patch_item_B = item_B.\
                    unfold(3, self.patch_size, self.patch_size).\
                    unfold(2, self.patch_size, self.patch_size).\
                    unfold(1, self.patch_size, self.patch_size)
                patch_item_B = patch_item_B.contiguous().view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                return {'A': patch_item_A,
                        'B': patch_item_B}


# Testing the code here
if __name__ == '__main__':
    dataroot = '/home/karthik/PycharmProjects/DLwithPyTorch/cycleGAN3D/datasets/mr2ct/'
    transforms_ = [transforms.Normalize((0.0,), (1.0,))]
    # since the image is already in [-1, 1] range, we normalize it by mu=0 and sigma=1. transforms.Normalize just
    # performs this operation: img = (img-mu)/sigma

    # there is doubt whether we should normalize an image that is already in [-1, 1] range. CHECK THIS
    data = ImageDataset(root_dir=dataroot, transform=transforms_, patches=False)
    # data = ImageDataset(root_dir=dataroot, transform=transforms_, mode='test', patches=False, one_side=True)
    data_loader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    temp = next(iter(data_loader))
    print(temp['A'].min(), temp['A'].max(), temp['A'].shape)
    print(temp['B'].shape)
    # output shape when patches=True -> [batch_size, num_patches, patch_size, patch_size, patch_size]
    # output shape when patches=False -> [batch_size, z, x, y]


