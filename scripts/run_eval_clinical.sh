# !/bin/bash

export CUDA_VISIBLE_DEVICES="4" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# --------------------------------------------

cls_mode="idh"
# cls_mode="1p_19q"
loss="bce"
metric="binary"
num_classes=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
slice_percentile=75

model="vit_base_clinical_prompt"
clini_info_style="bert"
embed_trainable="False"
decoder_depth="2"
clini_embed_token="cls"

save_dir="./exp/runs/idh/20240101/105947/"


# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

single_ckpt=$save_dir"train/checkpoint_best_loss.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

BEST_THRES=$(python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            best_thres=$BEST_THRES

python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            best_thres=$BEST_THRES