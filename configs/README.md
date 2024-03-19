# Train
## Binary 
python train.py gpus=2 

## Multiclass
python train.py trainer.gpus=3 cls_mode=subtype loss=ce model.net.num_classes=3 metric=multiclass



# Evaluation
## Binary 
python eval_internal.py gpus=3 task_name=val ckpt=/mai_nas/BYS/glioma/exp/train/runs/2023-11-14_14-35-45/checkpoint_best_auc.pth

