# !/bin/bash

export CUDA_VISIBLE_DEVICES="4" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# --------------------------------------------


cls_mode=$1
model=$2
slice_percentile=$3
save_dir=$4

metric="binary"
num_classes=1

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            best_thres=$BEST_THRES

python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            best_thres=$BEST_THRES

