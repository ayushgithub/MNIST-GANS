#!/usr/local/bin/python3
import argparse
import os

import idx2numpy
import numpy as np
from PIL import Image


def main(args):
    if args.dataset == 'MNIST':
        bin_files = [
            {'images': 'train-images-idx3-ubyte', 'labels': 'train-labels-idx1-ubyte'},
            {'images': 't10k-images-idx3-ubyte', 'labels': 't10k-labels-idx1-ubyte'}
        ]
        for bin_file in bin_files:
            images_file_path = os.path.join(args.source, bin_file['images'])
            labels_file_path = os.path.join(args.source, bin_file['labels'])
            if 'train' in bin_file['images']:
                destination_path = os.path.join(args.target, 'train')
            else:
                destination_path = os.path.join(args.target, 'test')
            os.makedirs(destination_path, exist_ok=True)
            images_arr = idx2numpy.convert_from_file(images_file_path)
            labels_arr = idx2numpy.convert_from_file(labels_file_path)
            num_images, _, _ = images_arr.shape
            file_name_padding = len(str(num_images))
            for i in range(num_images):
                img = Image.fromarray(images_arr[i])
                label = labels_arr[i]
                image_name = str(i).zfill(file_name_padding) + '-' + str(label) + '.jpg'
                destination = os.path.join(destination_path, image_name)
                img.save(destination, format='jpeg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True,
                        help='source directory where dataset is')
    parser.add_argument("-t", "--target", required=True,
                        help='target directory where processed images will be')
    parser.add_argument("-d", "--dataset", help="dataset to be processed")
    args = parser.parse_args()
    main(args)
