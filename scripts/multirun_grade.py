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


# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'False', '75', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'concat', 'True', '75', '7890'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade.sh', 'clinical_swint_small2', 'random', 'True', '2', 'self-attn', 'True', '75', '7890'])

subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '1234'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '2345'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '3456'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '4576'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '5678'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '6789'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '7890'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '1357'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '2468'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '3579'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_train_clinical_grade3.sh', 'clinical_swint_small2', 'bert', 'False', '2', 'self-attn', 'True', 'cls', '4680'])