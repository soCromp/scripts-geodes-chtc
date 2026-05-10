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

export CC=$(which gcc)
export CXX=$(which g++)

echo "expanding source code...  $(date)"
tar -xvzf geodes.tar.gz
cd geodes

start=$(date +%s)
mkdir data
tar -xzf /staging/groups/cs_geodes/cyclone/windmag/date/natlantic.tar.gz -C ./data

end=$(date +%s)
echo "data extraction took $((end - start)) seconds"

ls

# run scripts
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

python train_svd.py --dataset ./data/natlantic --channels 5 --num_train_epochs 10 --output_dir ./model --checkpoints_total_limit 1

tar -czvf /staging/groups/cs_geodes/${RUNNAME}_${PROCESS}.tar.gz ./model