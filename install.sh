#!/bin/bash

# 手动指定conda的路径（根据你的安装路径进行修改）
CONDA_PATH="$HOME/miniconda/bin/conda"

# 检测是否安装了conda
if [ -x "$CONDA_PATH" ]; then
    echo "Conda已安装在指定路径。"
    # 将conda路径添加到PATH中
    export PATH="$HOME/miniconda/bin:$PATH"
elif command -v conda &> /dev/null; then
    echo "Conda已安装在系统路径中。"
else
    echo "Conda未安装，正在安装Conda..."
    # 这里假设使用的是Miniconda安装方式，可以根据需要修改为Anaconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    echo "Conda安装完成。"
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# 检查是否存在名为guoneng的conda环境
if ! conda info --envs | grep -q "^guoneng"; then
    echo "guoneng环境不存在，正在创建..."
    conda create -n guoneng python==3.10.14 -y
    echo "guoneng环境创建完成。"
else
    echo "guoneng环境已存在。"
fi

# 激活guoneng环境
echo "正在激活guoneng环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate guoneng

echo "guoneng环境已激活。"


source ~/miniconda3/etc/profile.d/conda.sh
conda create -n guoneng python==3.10.14
conda activate guoneng
pip install -r requirements.txt