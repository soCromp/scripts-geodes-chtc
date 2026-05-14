#!/bin/bash

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
ModelName=$5

export CC=$(which gcc)
export CXX=$(which g++)

echo "expanding source code...  $(date)"
tar -xvzf geodes.tar.gz
cd geodes

start=$(date +%s)
mkdir data
tar -xzf /staging/groups/cs_geodes/cyclone/multivar/date/natlantic.tar.gz -C ./data
cp /staging/groups/cs_geodes/cyclone/multivar/date/channels.txt ./data/natlantic/channels.txt
end=$(date +%s)
echo "data extraction took $((end - start)) seconds"

ckpt_name=$(ls /staging/groups/cs_geodes | grep -E "^${ModelName}.tar\.gz$" | head -n 1)
ckpt="/staging/groups/cs_geodes/${ckpt_name}" # extra logic so we don't accidentally catch a prior sample’s tarball too
echo "expanding video model $ckpt..."
start=$(date +%s)
tar -xzf $ckpt -C .
end=$(date +%s)
echo "model extraction took $((end - start)) seconds"
ls

# run scripts
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
accelerate test --config_file accelerate_config_full.yaml
ACCELERATE_CONFIG_FILE=accelerate_config_full.yaml accelerate launch --num_processes 2 train_3d.py --train --continue \
    --epochs $Epochs --dataset ./data/natlantic/train --checkpoint_dir . --lr $LR --train_batch_size 1 \
    --save_image_epochs 100000 --save_model_epochs 1 --loss_fn huber --huber_delta 0.25 --name $ModelName
ls
tar -czvf /staging/groups/cs_geodes/${RUNNAME}_${PROCESS}_swa.tar.gz ./$ModelName*
