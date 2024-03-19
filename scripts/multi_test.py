
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


subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/test_multi_attmap.sh', 'idh', 'binary', '1', '/mai_nas/BYS/glioma/exp/runs/idh/20240220/165814/', '0.13863138854503632'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/test_multi_attmap.sh', '1p_19q', 'binary', '1', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240224/075347/', '0.2283521145582199'])
subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/test_multi_attmap2.sh', 'grade', 'multiclass', '3', '/mai_nas/BYS/glioma/exp/runs/grade/20240224/140726/'])
