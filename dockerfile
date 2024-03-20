# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# RUN mkdir -p /home/user/Multimodal_Transformer_Glioma && chown user:user /home/user/Multimodal_Transformer_Glioma

# RUN apt-get update
# RUN apt-get -y install libgl1-mesa-glx

# # USER user
# WORKDIR /home/user/Multimodal_Transformer_Glioma
# RUN chmod -R u+w /home/user/Multimodal_Transformer_Glioma

# ENV PATH="/home/user/.local/bin:${PATH}"

# RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
# RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# COPY --chown=user:user requirements.txt /home/user/Multimodal_Transformer_Glioma
# RUN pip install -r requirements.txt


FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /home/user/Multimodal_Transformer_Glioma && chown user:user /home/user/Multimodal_Transformer_Glioma

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /home/user/Multimodal_Transformer_Glioma
RUN chmod -R u+w /home/user/Multimodal_Transformer_Glioma

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install -U pip && python -m pip install pip-tools
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

COPY --chown=user:user requirements.txt /home/user/Multimodal_Transformer_Glioma
RUN pip install -r requirements.txt