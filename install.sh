#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n guoneng python==3.10.14
conda activate guoneng
pip install -r requirements.txt