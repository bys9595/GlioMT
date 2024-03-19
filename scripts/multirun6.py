
import subprocess
'''
model=$1
clini_info_style=$2
embed_trainable=$3
decoder_depth=$4
fusion_style=$5
clini=$6
slice_percentile=$7
randomseed=$8

'''


subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade2.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', '75', '2468'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade2.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', '75', '3579'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade2.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', '75', '4680'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade2.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', '75', '1470'])