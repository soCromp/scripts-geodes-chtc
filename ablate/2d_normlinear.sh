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

# run scripts
python train_2d.py --train --train --epochs 20 --dataset ./data/natlantic/train --checkpoint_dir . --name $RUNNAME \
    --lr 1e-5 --unet_block_out_channels 512,1024,2048 --save_image_epochs 100000 --save_model_epochs 100000 --train_batch_size 1 \
    --val_dataset ./data/satlantic/train --validation_epochs 10 --loss_fn l1 --normalize_all_linear
tar -czvf /staging/groups/cs_geodes/$RUNNAME\_$PROCESS.tar.gz ./$RUNNAME
