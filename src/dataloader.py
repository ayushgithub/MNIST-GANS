import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from tqdm import tqdm


class DataSetCustom(Dataset):
    """
        Custom dataloader for MNIST dataset. Keep in mind this stores the
        dataset in memory. Do not use it for big dataset.
        # TODO : Implement from disk dataloder too
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print('Loading dataset in memory....')
        self.root_dir = root_dir
        self.transform = transform
        images_path_list = glob.glob(self.root_dir+'/*.jpg')
        self.image_size = self._get_image_size(images_path_list[0])
        self.num_images = len(images_path_list)
        self.images = np.empty([self.num_images, self.image_size[0], self.image_size[1], self.image_size[2]])
        for idx in tqdm(range(self.num_images)):
            image_path = images_path_list[idx]
            img = np.array(Image.open(image_path))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            self.images[idx] = img

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx])

    def _get_image_size(self, image_path):
        img = Image.open(image_path)
        size = img.size
        if len(size) > 3:
            print("Number of channels in image is greater than 3. Please check")
            raise Exception
        if len(size) == 2:
            return (size[0], size[1], 1)
        return size


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class DataLoaderCustom():
    """ Custom dataloader with the given batch size and shuffle and num of workers """

    def __init__(self, root_dir, batch_size=50, shuffle=True, num_worker=1):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.num_worker = num_worker
        self.dataset = DataSetCustom(root_dir, ToTensor())
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_worker
        )
