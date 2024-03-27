# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

metric=$1
model=$2
cls_mode=$3
num_classes=$4
ckpt_path=$5 # .../checkpoint_best_auc.pth



eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
# internal validation
BEST_THRES=$(python eval_internal.py \
            ckpt=$ckpt_path \
            paths.save_dir=$save_dir \
            paths=internal_eval \
            metric=$metric \
            model=$model \
            cls_mode=$cls_mode \
            model.num_classes=$num_classes \
            paths.time_dir=$eval_time \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')


# external validation
python eval_external.py \
            ckpt=$ckpt_path \
            paths.save_dir=$save_dir \
            paths=TCGA \
            metric=$metric \
            model=$model \
            cls_mode=$cls_mode \
            model.num_classes=$num_classes \
            paths.time_dir=$eval_time \
            best_thres=$BEST_THRES


