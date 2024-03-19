# !/bin/bash

export CUDA_VISIBLE_DEVICES="1" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# --------------------------------------------

cls_mode="grade"
loss="ce"
metric="multiclass"
num_classes=3

# --------------------------------------------

slice_percentile=75

model="clinical_swint_small"
clini_info_style="bert"
embed_trainable="False"
decoder_depth="2"
clini_embed_token="cls"
age_cutoff=50

save_dir="/mai_nas/BYS/glioma/exp/runs/grade/20240109/001343/"

# Using Best AUC Model------------------------------------------------------------------------------------------------------------------------------------
single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
# ----------------------------------------------------------------------------------------------------------------------------------------

python eval_internal_clinical.py model=$model task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff 


python eval_external_clinical.py model=$model task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff 


python eval_external_clinical.py model=$model task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff 

