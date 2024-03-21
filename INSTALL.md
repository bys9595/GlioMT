# Installation
We provide installation instructions here.

### Clone our Repo
```
git clone https://github.com/bys9595/Multimodal_Transformer_Glioma.git
cd Multimodal_Transformer_Glioma
```


### Build Docker
The simplest way to use Multimodal Transformer is to build our dockerfile, which has contained all the needed dependencies. 

```
docker build -t mtg .
docker run -it --gpus all --net=host --pid=host --ipc=host -v .:/home/user/Multimodal_Transformer_Glioma/ --name mtg_container mtg:latest /bin/bash

docker run -it --gpus all --net=host --pid=host --ipc=host -v .:/home/user/Multimodal_Transformer_Glioma/ --name mtg_container mtg:latest /bin/bash

```


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train.py loss=bce metric=binary model=multimodal_vit_base cls_mode=idh model.num_classes=1 paths=train 
