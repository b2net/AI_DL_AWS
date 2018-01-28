#!/bin/bash
# Tested on Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-1aa7c063)
# https://aws.amazon.com/marketplace/pp/B06VSPXKDX
# Instance Type: p2.xlarge


### Installation of Caffe2 ###
# Remove the already existing source code of Caffe2, 
# since it has no module 'detectron'
echo Removing the old version of Caffe2..
cd /home/ubuntu/src
sudo rm -r caffe2

# Install the current version of Caffe2 (with module 'detectron')
echo Installing the new version of Caffe2..
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make && cd build && sudo make install

# Test the Caffe2 installation
echo Testing the Caffe2 installation:
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# Install Python dependencies
pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock

# Install COCO API
COCOAPI=/home/ubuntu/src/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI && sudo make install

### Installation of Detectron ###
# Clone the Detectron repository:
DETECTRON=/home/ubuntu/src/detectron
git clone https://github.com/facebookresearch/detectron $DETECTRON

# Set up Python modules:
cd $DETECTRON/lib && make

# Check that Detectron tests pass (e.g. for SpatialNarrowAsOp test):
echo Testing Detectron..
python2 $DETECTRON/tests/test_spatial_narrow_as_op.py

