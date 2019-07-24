#!/bin/bash

cwd=$(pwd)
mkdir data
mkdir data/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./data/raw
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./data/raw
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./data/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./data/raw
mkdir data/processed
cd data/raw
gunzip *
cd $cwd
pip3 install -r requirements.txt
python3 utils/extract_images.py -s ./data/raw -t ./data/processed -d MNIST --debug
