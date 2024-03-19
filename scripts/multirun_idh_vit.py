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

'''
model=$1
slice_percentile=$2
random_seed=$3 # 129 server, idh

# '''


# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'False', '75', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'concat', 'True', '75', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'random', 'True', '2', 'self-attn', 'True', '75', '7531'])

subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '4321'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '5432'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '6543'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '7654'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '8765'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '9876'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '0987'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '7531'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '8642'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '9753'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh.sh', 'clinical_vit_base2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '3579'])