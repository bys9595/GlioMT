# !/bin/bash

export CUDA_VISIBLE_DEVICES="7" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# --------------------------------------------

cls_mode=$1
loss="bce"
metric="binary"
num_classes=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
slice_percentile=75

model="clinical_vit_base_lrp"
clini_info_style="bert"
embed_trainable="False"
decoder_depth="2"
clini_embed_token="word"
age_cutoff=45
xai="vit_clinical_attmap"

save_dir="./exp/runs/1p_19q/20240130/021900/"


# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

BEST_THRES=$(python eval_internal_clinical_attmap.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff xai=$xai \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

python eval_external_clincical_attmap.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff xai=$xai \
            best_thres=$BEST_THRES

python eval_external_clincical_attmap.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            model.age_cutoff=$age_cutoff xai=$xai \
            best_thres=$BEST_THRES