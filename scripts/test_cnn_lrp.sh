# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# --------------------------------------------

cls_mode="idh"
# cls_mode="1p_19q"
loss="bce"
metric="binary"
num_classes=1

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------
model="resnet50_lrp"
slice_percentile=75
xai="cnn_gradcam"

save_dir="/mai_nas/BYS/glioma/exp/runs/idh/20231218/130434/"
BEST_THRES=0.10725543648004532

single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

# python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
#             metric=$metric \
#             model.num_classes=$num_classes \
#             model=$model \
#             cls_mode=$cls_mode \
#             data.slice_percentile=$slice_percentile xai=$xai \
#             best_thres=$BEST_THRES

# python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
#             metric=$metric \
#             model.num_classes=$num_classes \
#             model=$model \
#             cls_mode=$cls_mode \
#             data.slice_percentile=$slice_percentile xai=$xai \
#             best_thres=$BEST_THRES


xai="cnn_rcam"
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






cls_mode="1p_19q"
metric="binary"
num_classes=1

# --------------------------------------------
save_dir="/mai_nas/BYS/glioma/exp/runs/1p_19q/20231218/135713/"
xai="cnn_gradcam"


# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------
single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

echo $save_dir
echo $model
BEST_THRES=0.4665789008140564

# python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
#             metric=$metric \
#             model.num_classes=$num_classes \
#             model=$model \
#             cls_mode=$cls_mode \
#             data.slice_percentile=$slice_percentile xai=$xai \
#             best_thres=$BEST_THRES

# python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
#             metric=$metric \
#             model.num_classes=$num_classes \
#             model=$model \
#             cls_mode=$cls_mode \
#             data.slice_percentile=$slice_percentile xai=$xai \
#             best_thres=$BEST_THRES


xai="cnn_rcam"
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




cls_mode="grade"
metric="multiclass"
num_classes=3

# --------------------------------------------
end_num_of_exp=1 # num of exp - 1

save_dir="/mai_nas/BYS/glioma/exp/multiruns/grade/20231219/170541/"


# # Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------
# xai="cnn_gradcam"
# for i in $(seq $end_num_of_exp $end_num_of_exp)
# do
#     single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
#     job_num="/"$i
#     eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

#     echo $save_dir
#     echo $model

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


xai="cnn_rcam"
for i in $(seq $end_num_of_exp $end_num_of_exp)
do
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

    echo $save_dir
    echo $model

    python eval_external_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=$model \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile xai=$xai \

    python eval_external_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=$model \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile xai=$xai
done

