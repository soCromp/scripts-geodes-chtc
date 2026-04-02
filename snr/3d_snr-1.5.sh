#!/bin/bash

# Name snr -1.5 means no 2D SNR weighting and 3D snr weighting of 5

TZ="America/Chicago"
export PATH=/usr/sbin:$PATH
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
nvidia-smi

# arguments passed from sub file
RUNNAME=$1
PROCESS=$2
LR=$3
Epochs=$4
Image_Path=$5
Image_tag=$6

export CC=$(which gcc)
export CXX=$(which g++)

echo "expanding source code...  $(date)"
tar -xvzf geodes.tar.gz
cd geodes

start=$(date +%s)
mkdir data
tar -xzf /staging/groups/cs_geodes/cyclone/multivar/date/natlantic.tar.gz -C ./data
tar -xzf /staging/groups/cs_geodes/cyclone/multivar/date/satlantic.tar.gz -C ./data
cp /staging/groups/cs_geodes/cyclone/multivar/date/channels.txt ./data/natlantic/channels.txt
cp /staging/groups/cs_geodes/cyclone/multivar/date/channels.txt ./data/satlantic/channels.txt
end=$(date +%s)
echo "data extraction took $((end - start)) seconds"
echo "expanding image model $5... $(date)"
start=$(date +%s)
tar -xzf $5 -C .
end=$(date +%s)
echo "image model extraction took $((end - start)) seconds"
ls

# run scripts
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
accelerate test --config_file accelerate_config.yaml
ACCELERATE_CONFIG_FILE=accelerate_config.yaml accelerate launch --num_processes 2 train_3d.py --train \
    --epochs $4 --dataset ./data/natlantic/train --checkpoint_dir . --name ${RUNNAME}_${PROCESS} --lr $3 --img_model $6 --train_batch_size 1 \
    --save_image_epochs 100000 --save_model_epochs 100000 \
    --val_dataset ./data/satlantic/train --validation_epochs 4 --snr_gamma 5
ls
tar -czvf /staging/groups/cs_geodes/${RUNNAME}_${PROCESS}.tar.gz ./${RUNNAME}_${PROCESS}
