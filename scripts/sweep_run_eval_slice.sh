# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")


cls_mode="idh"
metric="binary"
num_classes=1

# --------------------------------------------
model="vit_base"
slice_percentile=(10 25 50 75 100)

save_dir="./exp/multiruns/idh/20231212/110140/" # must end with '/'
end_num_of_exp=9 # (num of exp - 1) 

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    for j in $(seq 0 4)
    do
        eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
        BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

        python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    best_thres=$BEST_THRES

        python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    best_thres=$BEST_THRES
    done
done

# Using Best Loss Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_loss.pth"
    job_num="/"$i
    for j in $(seq 0 4)
    do
        eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
        BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

        python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    best_thres=$BEST_THRES

        python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                    metric=$metric \
                    model.num_classes=$num_classes \
                    model=$model \
                    cls_mode=$cls_mode \
                    data.slice_percentile=${slice_percentile[j]} \
                    best_thres=$BEST_THRES
    done
done