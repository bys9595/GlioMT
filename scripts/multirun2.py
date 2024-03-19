import subprocess
'''
model=$1
clini_info_style=$2
embed_trainable=$3
decoder_depth=$4
fusion_style=$5
clini=$6
clini_embed_token=$7
random_seed=$8 # 129 server, idh
slice_percentile=$9
'''

'''
model=$1
slice_percentile=$2
random_seed=$3 # 129 server, idh

# '''

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_idh3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_1p19q3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '7890'])


subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '75', '7890'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '75', '7890'])


# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'resnet50', '0.1', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'resnet50', '10', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'resnet50', '25', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'resnet50', '50', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'resnet50', '100', '7531'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'vit_base', '0.1', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'vit_base', '10', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'vit_base', '25', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'vit_base', '50', '7531'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_idh.sh', 'vit_base', '100', '7531'])




# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'resnet50', '0.1', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'resnet50', '10', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'resnet50', '25', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'resnet50', '50', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'resnet50', '100', '3456'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'vit_base', '0.1', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'vit_base', '10', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'vit_base', '25', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'vit_base', '50', '3456'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_1p19q.sh', 'vit_base', '100', '3456'])





# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '0.1', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '10', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '25', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '50', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'resnet50', '100', '7890'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '0.1', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '10', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '25', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '50', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_grade.sh', 'vit_base', '100', '7890'])
