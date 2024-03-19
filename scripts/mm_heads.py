import subprocess
'''
model=$1
clini_info_style=$2
embed_trainable=$3
decoder_depth=$4
fusion_style=$5
clini=$6
slice_percentile=$7
'''

subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh2.sh', 'clinical_swint_small', 'bert', 'False', '2', 'self-attn', 'True', '75', '4'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh2.sh', 'clinical_swint_small', 'bert', 'False', '2', 'self-attn', 'True', '75', '8'])
