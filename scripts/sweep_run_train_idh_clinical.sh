# !/bin/bash

export CUDA_VISIBLE_DEVICES="1"
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# ---------------------------------------
random_seed="3579" # 129 server, idh

cls_mode="idh"
loss="bce"
metric="binary"
num_classes=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# custom_pretrained_pth="/mai_nas/BYS/mae/output_dir/mae_ch4_10percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_25percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_50percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_75percent/checkpoint-799.pth"

model="clinical_vit_base,clinical_swint_small,clinical_swint_base"

embed_trainable="False"
clini_embed_token="word"
clini="True"


# --------------------------------------------T-R-A-I-N-I-N-G-------------------------------------------------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python train_clinical.py -m model=$model \
                    loss=$loss metric=$metric \
                    model.num_classes=$num_classes \
                    model.embed_trainable=$embed_trainable \
                    model.clini_embed_token=$clini_embed_token \
                    model.clini=$clini \
                    paths=train paths.time_dir=$now trainer.end_epoch=1 \
                    cls_mode=$cls_mode random_seed=$random_seed                     

# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE

save_dir="./exp/multiruns/"$cls_mode"/"$now"/"

end_num_of_exp=5 # num of exp - 1
model=(clinical_vit_base clinical_swint_small clinical_swint_base clinical_vit_base clinical_swint_small clinical_swint_base)
# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
    BEST_THRES=$(python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model=${model[i]} \
                model.num_classes=$num_classes \
                cls_mode=$cls_mode \
                model.embed_trainable=$embed_trainable \
                model.clini_embed_token=$clini_embed_token \
                model.clini=$clini \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

    python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model=${model[i]} \
                model.num_classes=$num_classes \
                cls_mode=$cls_mode \
                model.embed_trainable=$embed_trainable \
                model.clini_embed_token=$clini_embed_token \
                model.clini=$clini \
                best_thres=$BEST_THRES

    python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model=${model[i]} \
                model.num_classes=$num_classes \
                cls_mode=$cls_mode \
                model.embed_trainable=$embed_trainable \
                model.clini_embed_token=$clini_embed_token \
                model.clini=$clini \
                best_thres=$BEST_THRES
done