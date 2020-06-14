# mangogogo

## Prerequisites

* Linux *(tested on Ubuntu 16.04)*
* Python 3 *(tested on python 3.7)*
* NVIDIA GPU + CUDA & CuDNN *(tested on CUDA 10.0 & CuDNN 7.6)*

## For Users with Nvidia-Docker 
First, install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart).

We provide an Dockerfile in this repository. 

* Clone this repository.
```
git clone https://github.com/IanYeung/mangogogo
cd mangogogo
```

* Build the image.
```
docker build -t mangogogo -f Dockerfile .
```

* Start the image (please replace "/data" before the ":" sign with the data root on your local machine).
```
docker run --gpus all -it -v /data:/data --ipc=host mangogogo
```

* You are now good to go.

## For Users without Nvidia-Docker 
* Clone this repository

```
git clone https://github.com/IanYeung/mangogogo
cd mangogogo
```

* Install dependencies

```
pip install -r requirement.txt
```

* Compile the Deformable Convolution module. We employ the [Deformable Convolution](https://arxiv.org/abs/1703.06211) (dcn) implementation from [mmdetection](https://github.com/open-mmlab/mmdetection). Please first compile it.

```
cd codes/models/archs/dcn
python setup.py develop
cd ../../../..
```

## License
This project is released under the Apache 2.0 license.
