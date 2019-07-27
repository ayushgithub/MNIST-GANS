import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from dataloader import DataLoaderCustom
from matplotlib import pyplot as plt
from model import Discriminator, Generator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEVICE = torch.device("cpu")


def flatten(tensor):
    # assuming first dimension is batch dimension
    return tensor.view(tensor.size(0), -1)


def zeros(batch_size, args):
    ret_tensor = Variable(torch.zeros(batch_size))
    if use_cuda(args):
        ret_tensor.to(DEVICE)
    return ret_tensor


def ones(batch_size, args):
    ret_tensor = Variable(torch.ones(batch_size))
    if use_cuda(args):
        ret_tensor.to(DEVICE)
    return ret_tensor


def use_cuda(args):
    return not args.no_cuda and torch.cuda.is_available()


def noise(batch_size, args):
    # noise from gaussian distribution of 0 mean and 1 std
    ret_tensor = Variable(torch.randn(batch_size, 100))
    if use_cuda(args):
        return ret_tensor.to(DEVICE)
    return ret_tensor


def train_discriminator(real_images, fake_images, discriminator_net, discriminator_optim, args):
    batch_size = args.batch_size
    criterion = nn.BCELoss()

    discriminator_optim.zero_grad()

    out_real = discriminator_net(real_images)
    loss_real = criterion(out_real, ones(batch_size, args))
    loss_real.backward()

    out_fake = discriminator_net(fake_images)
    loss_fake = criterion(out_fake, zeros(batch_size, args))
    loss_fake.backward()

    discriminator_optim.step()
    return loss_fake+loss_real, out_real, out_fake


def train_generator(fake_images, discriminator_net, generator_optim, args):
    batch_size = args.batch_size
    generator_optim.zero_grad()
    out_fake = discriminator_net(fake_images)
    criterion = nn.BCELoss()
    loss = criterion(out_fake, ones(batch_size, args))
    loss.backward()
    generator_optim.step()
    return loss


def get_generated_images(images_flattened):
    images_flattened = images_flattened.mul(0.5).add(0.5).mul(255) # images now between [0,1]
    batch_size, image_size_flattened = images_flattened.size() 
    width = height = math.sqrt(image_size_flattened)
    return image_size_flattened.view(batch_size, width, height).numpy()


def sample(generator_net, test_noise, batch_no, epoch_no, image_save_folder, args):
    out_generated = generator_net(test_noise)
    images = get_generated_images(out_generated)
    batch_size = images.size(0)
    grid_size = math.sqrt(batch_size)

    fig = plt.figure()
    for i in range(batch_size):
        fig.add_subplot(grid_size, grid_size, i+1)
        plt.imshow(images[i])
    image_save_path = os.path.join(image_save_folder, epoch_no+'_'+batch_no+'.jpg')
    fig.savefig(image_save_path)
    plt.close()


def train(discriminator_net, generator_net, train_loader, args):
    image_save_folder = os.path.join(args.image_save_folder, args.uid)
    model_save_folder = os.path.join(args.model_save_folder, args.uid)
    os.makedirs(image_save_folder, exist_ok=True)
    os.makedirs(model_save_folder, exist_ok=True)

    discriminator_optim = optim.Adam(
        discriminator_net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon
    )
    generator_optim = optim.Adam(
        generator_net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon
    )
    batch_size = args.trainbs
    test_noise = noise(args.test_set_size, args)
    for epoch in range(args.epoch):
        for batch_no, real_images in enumerate(train_loader.dataloader):
            # Train Discriminator
            real_images = flatten(real_images)
            fake_images = generator_net(noise(batch_size, args)).detach()
            discriminator_optim.zero_grad()
            predictions = discriminator_net()
            discriminator_loss, real_out, fake_out = train_discriminator(
                real_images,
                fake_images,
                discriminator_net,
                discriminator_optim,
                args
            )
            fake_images = generator_net(noise(batch_size, args))
            generator_loss = train_generator(
                fake_images,
                discriminator_net,
                generator_optim,
                args
            )
            if batch_no % args.log_interval == 0:
                print("discriminator loss: {}, generator loss: {}".format(
                    discriminator_loss, generator_loss
                ))
            if batch_no % args.test_frequency == 0:
                sample(generator_net, test_noise, batch_no, epoch, image_save_folder, args)



def main(args):
    # create some default directories
    os.makedirs(os.path.join(ROOT_DIR, 'generated_images'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'model'), exist_ok=True)

    torch.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if use_cuda(args) else "cpu")
    train_loader = DataLoaderCustom(args.train_dataset, batch_size=args.trainbs)
    discriminator_net = Discriminator()
    generator_net = Generator()
    if use_cuda(args):
        discriminator_net.to(DEVICE)
        generator_net.to(DEVICE)
    train(discriminator_net, generator_net, train_loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainbs', type=int, default=64,
                        help='Batch size while training')
    parser.add_argument('--testbs', type=int, default=64,
                        help='Batch size while testing')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning Rate')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs to train the network')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for adam optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon for adam')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train_dataset', default=os.path.join(ROOT_DIR, 'data/processed/train'),
                        help='Path to the training images directory')
    parser.add_argument('--test_dataset', default=os.path.join(ROOT_DIR, 'data/processed/test'))
    parser.add_argument('--no_cuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--model_save_folder', default=os.path.join(ROOT_DIR, 'model'),
                        help='Directory to save models')
    parser.add_argument('--test_set_size', type=int, default=16,
                        help='Size of test set for generating images')
    parser.add_argument('--test_frequency', type=int, default=100,
                        help='How frequently to generate test images')
    parser.add_argument('--image_save_folder', default=os.path.join(ROOT_DIR, 'generated_images'),
                        help='Folder to save the generated images')
    parser.add_argument('--uid', type=str, required=True,
                        help='Unique identifier for the run')
    args = parser.parse_args()
    main(args)
