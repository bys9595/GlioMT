
import subprocess
'''
cls_mode=$1
metric=$2
num_classes=$3
save_dir=$4


'''


subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/test_multi_attmap_val2.sh', 'grade', 'multiclass', '3', '/mai_nas/BYS/glioma/exp/runs/grade/20240229/200243/', 'cls'])
