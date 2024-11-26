# Installation
We provide installation instructions here.

### Clone our Repo
```
git clone https://github.com/bys9595/GlioMT.git
cd GlioMT
```


### Build Docker
The simplest way to use GlioMT is to build our dockerfile, which has contained all the needed dependencies. 

```
docker build -t gliomt .
docker run -it --gpus all --net=host \
    --pid=host --ipc=host \
    -v ./:/home/user/GlioMT \
    -v {data_path}:/home/user/GlioMT/data \
    --name gliomt_container gliomt:latest /bin/bash
```
