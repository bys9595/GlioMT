# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")


# --------------------------------------------

cls_mode="grade"
loss="ce"
metric="multiclass"
num_classes=3
# --------------------------------------------

model=(resnet18 resnet50 densenet121 vit_base swint_base)
# model=(efficientnetv2_m efficientnetv2_l vit_tiny vit_small vit_base swint_tiny swint_small swint_base)

save_dir="./exp/multiruns/grade/20240104/174928/"

end_num_of_exp=4 # (num of exp) - 1
slice_percentile="50"


# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
    python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile

    python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \

    python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile
done

# Using Best Loss Model ---------------------------------------------------------------------------------------------------------------------------------------
for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_loss.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
    python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile

    python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile

    python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile
done