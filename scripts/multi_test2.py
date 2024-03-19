
import subprocess
'''
cls_mode=$1
model=$2
slice_percentile=$3
save_dir=$4



'''

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '0.1', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/161546/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '10', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/172337/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '25', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/181255/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '50', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/185927/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '75', '/mai_nas/BYS/glioma/exp/runs/idh/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'resnet50', '100', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/201153/'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '0.1', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/214955/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '10', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/223634/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '25', '/mai_nas/BYS/glioma/exp/runs/idh/20240308/235301/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '50', '/mai_nas/BYS/glioma/exp/runs/idh/20240309/010436/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '75', '/mai_nas/BYS/glioma/exp/runs/idh/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', 'idh', 'vit_base', '100', '/mai_nas/BYS/glioma/exp/runs/idh/20240309/021415/'])



# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '0.1', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/034404/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '10', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/044615/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '25', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/060338/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '50', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/070801/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '75', '/mai_nas/BYS/glioma/exp/runs/1p_19q/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'resnet50', '100', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/075801/'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '0.1', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/085239/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '10', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/093939/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '25', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/110715/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '50', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/122837/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '75', '/mai_nas/BYS/glioma/exp/runs/1p_19q/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval.sh', '1p_19q', 'vit_base', '100', '/mai_nas/BYS/glioma/exp/runs/1p_19q/20240309/133738/'])



# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '0.1', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/145817/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '10', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/155442/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '25', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/164137/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '50', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/175346/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '75', '/mai_nas/BYS/glioma/exp/runs/grade/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'resnet50', '100', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/185314/'])

# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '0.1', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/195542/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '10', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/211024/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '25', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/222955/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '50', '/mai_nas/BYS/glioma/exp/runs/grade/20240309/231338/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '75', '/mai_nas/BYS/glioma/exp/runs/grade/xxxx/xxxx/'])
# subprocess.run(['bash', '/mai_nas/BYS/glioma/scripts/run_eval_grade.sh', 'grade', 'vit_base', '100', '/mai_nas/BYS/glioma/exp/runs/grade/20240310/000127/'])


# import os
# import json
# import numpy as np
# import yaml


# root_dir = '/mai_nas/BYS/glioma/exp/runs/grade/20240309/' # grade vit
# metric_name = 'Sensitivity'
# # metric_name = 'Specificity'

# outdict = {'slice_percentile': [], 'model': [], 'cls_mode' : [], 'dir' : []}

# for (root, dirs, files) in os.walk(root_dir):
#     if 'checkpoint_best_auc.pth' in files:
#         with open(os.path.join(root, '.hydra', 'overrides.yaml')) as yaml_file:
#             yaml_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
            
#             outdict['dir'].append(root.split('/')[-2])
#             for data in yaml_data:
#                 if 'model=' in data:
#                     outdict['model'].append(data.split('=')[-1])
#                 elif 'cls_mode=' in data:
#                     outdict['cls_mode'].append(data.split('=')[-1])
#                 elif 'data.slice_percentile=' in data:
#                     outdict['slice_percentile'].append(data.split('=')[-1])

# import pandas as pd
# with pd.ExcelWriter('test.xlsx') as writer:
#     df0 = pd.DataFrame(outdict)
#     df0.to_excel(writer, startcol=0)  
