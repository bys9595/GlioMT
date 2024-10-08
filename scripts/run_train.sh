# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

loss=$1
metric=$2
model=$3
cls_mode=$4
num_classes=$5
slice_percentile=$6
train_kwargs=$7

random_seed=1234

HYDRA_FULL_ERROR=1 python train.py \
                    loss=$loss metric=$metric model=$model \
                    cls_mode=$cls_mode model.num_classes=$num_classes \
                    paths=train paths.time_dir=$now random_seed=$random_seed \
                    data.slice_percentile=$slice_percentile \
                    $train_kwargs


# -----------------------------------------------------------------------------------------------------------------------
# If you want to evaluate the model, please activate the code below
# -----------------------------------------------------------------------------------------------------------------------

save_dir="./exp/runs/"$cls_mode"/"$now"/"

ckpt_dir=$save_dir"/train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

if [ "$cls_mode" == "grade" ];then 
    # internal validation
    python eval_internal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=internal_eval \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                data.slice_percentile=$slice_percentile

    # external validation
    python eval_external.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=TCGA \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                data.slice_percentile=$slice_percentile

else
    # internal validation
    BEST_THRES=$(python eval_internal.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=internal_eval \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                data.slice_percentile=$slice_percentile \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')


    # external validation
    python eval_external.py \
                ckpt=$ckpt_dir \
                paths.save_dir=$save_dir \
                paths=TCGA \
                metric=$metric \
                model=$model \
                cls_mode=$cls_mode \
                model.num_classes=$num_classes \
                paths.time_dir=$eval_time \
                data.slice_percentile=$slice_percentile \
                best_thres=$BEST_THRES
fi

