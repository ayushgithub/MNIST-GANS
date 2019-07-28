import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from dataloader import DataLoaderCustom
from model import Discriminator, Generator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEVICE = torch.device("cpu")


def flatten(tensor):
    # assuming first dimension is batch dimension
    return tensor.view(tensor.size(0), -1)


def reshape_images(images):
    image_size = math.ceil(math.sqrt(images.size(1)))
    return images.view(images.size(0), 1, image_size, image_size)


def zeros(batch_size, args):
    ret_tensor = Variable(torch.zeros(batch_size, 1))
    if use_cuda(args):
        ret_tensor.to(DEVICE)
    return ret_tensor


def ones(batch_size, args):
    ret_tensor = Variable(torch.ones(batch_size, 1))
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
    batch_size = args.trainbs
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
    batch_size = args.trainbs
    generator_optim.zero_grad()
    out_fake = discriminator_net(fake_images)
    criterion = nn.BCELoss()
    loss = criterion(out_fake, ones(batch_size, args))
    loss.backward()
    generator_optim.step()
    return loss


def sample(generator_net, test_noise):
    with torch.no_grad():
        out_generated = generator_net(test_noise)
    images = reshape_images(out_generated)
    return utils.make_grid(images, normalize=True, scale_each=True)


def train(discriminator_net, generator_net, train_loader, args):
    print('Starting Training....')
    logs_save_folder = os.path.join(args.logs_save_folder, args.uid)
    model_save_folder = os.path.join(args.model_save_folder, args.uid)
    os.makedirs(logs_save_folder, exist_ok=True)
    os.makedirs(model_save_folder, exist_ok=True)
    writer = SummaryWriter(logs_save_folder)

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
    num_batches = len(train_loader.dataloader)
    test_noise = noise(args.test_set_size, args)
    for epoch in range(args.epoch):
        for batch_no, real_images in enumerate(train_loader.dataloader):
            # Train Discriminator
            real_images = flatten(real_images)
            fake_images = generator_net(noise(batch_size, args)).detach()
            discriminator_optim.zero_grad()
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
                print("Epoch: {}, batch_no: {}, discriminator loss: {}, generator loss: {}".format(
                    epoch, batch_no, discriminator_loss, generator_loss
                ))
                tb_step = epoch*num_batches + batch_no
                writer.add_scalar('Generator Loss', generator_loss, tb_step)
                writer.add_scalar('Discriminator Loss', discriminator_loss, tb_step)
            if batch_no % args.test_frequency == 0:
                tb_step = epoch*num_batches + batch_no
                generated_images = sample(generator_net, test_noise)
                writer.add_image('generated_images', generated_images, tb_step)
    
    writer.close()


def main(args):
    # create some default directories
    os.makedirs(os.path.join(ROOT_DIR, 'logs'), exist_ok=True)
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
    parser.add_argument('--logs_save_folder', default=os.path.join(ROOT_DIR, 'logs'),
                        help='Folder to save the generated images')
    parser.add_argument('--uid', type=str, required=True,
                        help='Unique identifier for the run')
    args = parser.parse_args()
    main(args)
