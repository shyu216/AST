#!/bin/bash
#SBATCH --mail-user=shyu0@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH -p gpu_24h
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
##SBATCH --mem=48000
#SBATCH --job-name="ast"
#SBATCH --output=./log_%j.txt

# set -x
# # comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=audioset
imagenetpretrain=True
audiosetpretrain=True


bal=bal
lr=1e-5
epoch=30
freqm=24
timem=96
mixup=0
fstride=10
tstride=10
batch_size=12
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85
wa_start=1
wa_end=5


tr_data=./data/datafiles/train.json
va_data=./data/datafiles/eval.json
te_data=./data/datafiles/test.json

## mean:  -0.0003213350119944467
## std:  0.0034993222207247835
dataset_mean=-0.000321335
dataset_std=0.003499322
audio_length=512
noise=False

metrics=mAP
loss=BCE
warmup=False
wa=True

exp_dir=./exp/ast-attempt7-acc-mean-std
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py \ 
-w 0 \
--model ${model} \
--dataset ${dataset} \
--data-train ${tr_data} \
--data-val ${va_data} --exp-dir $exp_dir \
--data-eval ${te_data} \
--label-csv ./data/class_labels_indices.csv \
--n_class 88 \
--lr $lr \
--n-epochs ${epoch} \
--batch-size $batch_size \
--save_model True \
--freqm $freqm --timem $timem \
--mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--imagenet_pretrain $imagenetpretrain \
--audioset_pretrain ${audiosetpretrain} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} \
--lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end}
