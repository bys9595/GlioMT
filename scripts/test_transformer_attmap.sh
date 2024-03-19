# !/bin/bash

export CUDA_VISIBLE_DEVICES="1" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")


cls_mode="idh"
metric="binary"
num_classes=1

# --------------------------------------------

slice_percentile=75
end_num_of_exp=8 # num of exp - 1
model=vit_base_lrp
xai=vit_attmap

save_dir="/mai_nas/BYS/glioma/exp/multiruns/idh/20231214/025453/"

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq $end_num_of_exp $end_num_of_exp)
do
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

    echo $save_dir
    echo $model
    # BEST_THRES=$(python eval_internal_attmap.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval \
    #             paths.time_dir=$eval_time metric=$metric \
    #             model.num_classes=$num_classes model=$model cls_mode=$cls_mode \
    #             data.slice_percentile=$slice_percentile xai=$xai \
    #             | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')
    BEST_THRES=0.2669496536254883

    python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=$model \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile xai=$xai \
                best_thres=$BEST_THRES

    python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=$model \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile xai=$xai \
                best_thres=$BEST_THRES
done












cls_mode="1p_19q"
metric="binary"
num_classes=1

# --------------------------------------------

slice_percentile=75

save_dir="/mai_nas/BYS/glioma/exp/runs/1p_19q/20231218/151413/"
model=vit_base_lrp
xai=vit_attmap


# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------
single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

echo $save_dir
echo $model
# BEST_THRES=$(python eval_internal_attmap.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval \
#             paths.time_dir=$eval_time metric=$metric \
#             model.num_classes=$num_classes model=$model cls_mode=$cls_mode \
#             data.slice_percentile=$slice_percentile xai=$xai \
#             | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')
BEST_THRES=0.6260897517204285

python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile xai=$xai \
            best_thres=$BEST_THRES

python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile xai=$xai \
            best_thres=$BEST_THRES















# cls_mode="grade"
# metric="multiclass"
# num_classes=3

# # --------------------------------------------

# slice_percentile=75
# end_num_of_exp=4 # num of exp - 1

# save_dir="/mai_nas/BYS/glioma/exp/multiruns/grade/20231219/171801/"
# model=vit_base_lrp
# xai=vit_attmap


# # Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

# for i in $(seq $end_num_of_exp $end_num_of_exp)
# do
#     single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
#     job_num="/"$i
#     eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

#     echo $save_dir
#     echo $model
#     # BEST_THRES=$(python eval_internal_attmap.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval \
#     #             paths.time_dir=$eval_time metric=$metric \
#     #             model.num_classes=$num_classes model=$model cls_mode=$cls_mode \
#     #             data.slice_percentile=$slice_percentile xai=$xai \
#     #             | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

#     python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
#                 metric=$metric \
#                 model.num_classes=$num_classes \
#                 model=$model \
#                 cls_mode=$cls_mode \
#                 data.slice_percentile=$slice_percentile xai=$xai \

#     python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
#                 metric=$metric \
#                 model.num_classes=$num_classes \
#                 model=$model \
#                 cls_mode=$cls_mode \
#                 data.slice_percentile=$slice_percentile xai=$xai
# done

