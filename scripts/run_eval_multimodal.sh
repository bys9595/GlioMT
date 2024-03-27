# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

metric=$1
model=$2
cls_mode=$3
num_classes=$4
save_dir=$5 # ./exp/runs/idh/20240101/090000
eval_kwargs=$6

ckpt_dir=$save_dir"/train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

if [ "$cls_mode" == "grade" ];then 
    # internal validation
    python eval_internal_multimodal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=internal_eval \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                $eval_kwargs

    # external validation
    python eval_external_multimodal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=TCGA \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                $eval_kwargs
else
    # internal validation
    BEST_THRES=$(python eval_internal_multimodal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=internal_eval \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                $eval_kwargs \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')


    # external validation
    python eval_external_multimodal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=TCGA \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                best_thres=$BEST_THRES \
                $eval_kwargs
fi


