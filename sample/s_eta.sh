#!/bin/bash

TZ="America/Chicago"
export PATH=/usr/sbin:$PATH
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
nvidia-smi

# arguments passed from sub file
RUNNAME=$1
CLUSTER=$2
PROCESS=$3
Dataset=$4
Channelfile=$5
Split=$6
ModelName=$7
Eta=$8

StartSample=$((PROCESS * 64))
EndSample=$(((PROCESS + 1) * 64))
if [ "$EndSample" -gt 900 ]; then
    EndSample=900
fi

echo "$RUNNAME $PROCESS: Sample indices $StartSample inclusive to $EndSample exclusive"

echo "expanding source code...  $(date)"
tar -xzf geodes.tar.gz
cd geodes

Dataname=$(basename $Dataset .tar.gz)
start=$(date +%s)
mkdir data
tar -xzf $Dataset -C ./data
end=$(date +%s)
echo "data extraction took $((end - start)) seconds. Dataset name is $Dataname"
cp $Channelfile ./data/natlantic/

ckpt_name=$(ls /staging/groups/cs_geodes | grep -E "^${ModelName}.tar\.gz$" | head -n 1)
ckpt="/staging/groups/cs_geodes/${ckpt_name}" # extra logic so we don't accidentally catch a prior sample’s tarball too
echo "expanding video model $ckpt..."
start=$(date +%s)
tar -xzf $ckpt -C .
end=$(date +%s)
echo "model extraction took $((end - start)) seconds"

ls

# run scripts
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python train_3d.py --dataset ./data/natlantic/$Split --checkpoint_dir . --name $ModelName --eval_batch_size 32 \
    --start_idx $StartSample --end_idx $EndSample --eta $Eta
ls
tar -czvf /staging/cromp/$RUNNAME\_$CLUSTER\_$PROCESS.tar.gz ./$ModelName/samples
