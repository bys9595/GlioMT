# !/bin/bash

export CUDA_VISIBLE_DEVICES="3" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# ---------------------------------------
random_seed="3579" # 51 server, 1p/19q

cls_mode="1p_19q"
loss="bce"
metric="binary"
num_classes=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# custom_pretrained_pth="/mai_nas/BYS/mae/output_dir/mae_ch4_75percent_ACS/checkpoint-299.pth"
# global_pool="avg"
slice_percentile=75

model="resnet50"
# model="resnet50_lrp,vit_base_lrp,resnet18,resnet50,resnet152,densenet121,densenet201,efficientnet_b0,efficientnet_b2,efficientnet_b4,efficientnetv2_m,efficientnetv2_l,vit_tiny,vit_small,vit_base,swint_tiny,swint_small,swint_base,vit_large,swint_large"

# --------------------------------------------T-R-A-I-N-I-N-G-------------------------------------------------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python train.py \
                    loss=$loss metric=$metric model=$model \
                    cls_mode=$cls_mode random_seed=$random_seed model.num_classes=$num_classes paths=train \
                    paths.time_dir=$now data.slice_percentile=$slice_percentile \
                    model.pretrained=True
                    # model.global_pool=$global_pool \
                    # custom_pretrained_pth=$custom_pretrained_pth

# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

save_dir="./exp/runs/"$cls_mode"/"$now"/"

single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile \
            best_thres=$BEST_THRES

python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            best_thres=$BEST_THRES