FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y wget curl zip git gcc g++ vim tmux ffmpeg
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda3/bin:${PATH}
RUN conda update -y conda

RUN mkdir /workspace && cd /workspace && mkdir mgtv
COPY . /workspace/mgtv
RUN pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN cd /workspace/mgtv && pip install -r requirements.txt
RUN cd /workspace/mgtv/codes/models/archs/dcn && python setup.py develop

WORKDIR /workspace
#RUN chmod 777 /workspace/mgtv/codes/test.sh
#CMD ["/bin/bash", "-c", "/workspace/mgtv/codes/test.sh"]