
CUDA_VISIBLE_DEVICES=0 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade resnet50 0.1 /mai_nas/BYS/glioma/exp/runs/grade/20240309/145817/
CUDA_VISIBLE_DEVICES=1 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade resnet50 10 /mai_nas/BYS/glioma/exp/runs/grade/20240309/155442/
CUDA_VISIBLE_DEVICES=2 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade resnet50 25 /mai_nas/BYS/glioma/exp/runs/grade/20240309/164137/
CUDA_VISIBLE_DEVICES=3 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade resnet50 50 /mai_nas/BYS/glioma/exp/runs/grade/20240309/175346/
CUDA_VISIBLE_DEVICES=4 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade resnet50 100 /mai_nas/BYS/glioma/exp/runs/grade/20240309/185314/

CUDA_VISIBLE_DEVICES=1 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade vit_base 0.1 /mai_nas/BYS/glioma/exp/runs/grade/20240309/195542/
CUDA_VISIBLE_DEVICES=0 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade vit_base 10 /mai_nas/BYS/glioma/exp/runs/grade/20240309/211024/
CUDA_VISIBLE_DEVICES=1 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade vit_base 25 /mai_nas/BYS/glioma/exp/runs/grade/20240309/222955/
CUDA_VISIBLE_DEVICES=0 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade vit_base 50 /mai_nas/BYS/glioma/exp/runs/grade/20240309/231338/
CUDA_VISIBLE_DEVICES=3 bash /mai_nas/BYS/glioma/scripts/run_eval_grade.sh grade vit_base 100 /mai_nas/BYS/glioma/exp/runs/grade/20240310/000127/
