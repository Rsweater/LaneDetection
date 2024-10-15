# install.sh 脚本代码如下
#!/usr/bin/env bash

# Copyright (c) 2018 Baowang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# author:   Baowang Chen <bw_chen1223@foxmail.com>
# version:  2024.08
# date:     June 15, 2024

# This script is used to install the LaneDetection package.

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure install conda is installed.
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install conda first."
    exit 1
fi

# If you have not cloned the LaneDetection_mm project, input the path to clone it.
# elif, input the path of the LaneDetection_mm project.
# cd to the LaneDetection_mm project.
read -p "Have you cloned the LaneDetection project? (y/n) " answer
if [ "$answer" == "n" ]; then
    read -p "Please input the path to clone the LaneDetection project: " path
    git clone https://github.com/Rsweater/LaneDetection.git $path
    cd $path/LaneDetection
else
    read -p "Please input the path of the LaneDetection project: " path
    cd $path
fi

read -p "Please input the name of the virtual environment: " env_name
printf "Creating the virtual environment...\n"
conda create -n $env_name python=3.8 -y
printf "The virtual environment has been created successfully.\n"

# View and activate the virtual environment.
conda env list
conda activate $env_name

# Install the required packages.
printf "Installing the required packages...\n"
printf "Please wait for a moment...\n"
printf "Fistly, we run the command 'conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia'
to install the pytorch package with CUDA 11.1 support.\n"
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
printf "The pytorch package has been installed successfully.\n"
printf "Secondly, we install mmdetection==2.25.3 and mmcv-full==1.5.1 packages.\n"
pip install -U openmim
mim install mmcv-full==1.5.1
pip install mmdet==2.25.3
printf "The mmdetection==2.25.3 and mmcv-full==1.5.1 packages have been installed successfully.\n"
printf "Finally, we install the other required packages.\n"
pip install -r requirements.txt
printf "The other required packages have been installed successfully.\n"

# Complie ops
printf "Compiling the ops...\n"
cd libs/models/layers/nms/ # 确保本地cuda版本与conda中一致
python setup.py install
cd ../../../../ # TODO: 直接导入
printf "The ops have been compiled successfully.\n"

# Setting the environment variable for the LaneDetectio project.
PYTHONPATH="$(dirname $0)":$PYTHONPATH

# Done.
printf "The installation is done.\n"



