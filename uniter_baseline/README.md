## Steps to reproduce

First, `git clone https://github.com/ajd12342/why-winoground-hard.git`, then download winoground dataset and place the winoground images under `why-winoground-hard/dataset/images`

### Generate RoI features
The [codebase](https://github.com/ChenRocks/UNITER/tree/master) is rather old and currently no information is released for winoground. so we have to piece together the information ourselves. All of the following is tested on `AWS g4dn.xlarge` in Oct 2023 using the Deep Learning base AMI.

To generate features with BUTD, the original implementation uses Caffe.

The UNITER author has an open source version [here](https://github.com/ChenRocks/BUTD-UNITER-NLVR2.git) that shows an example to generate new bounding box features on a new dataset. You can just pull his docker to run it. 
To get an interactive docker, do this
```
CUDA_VISIBLE_DEVICES=0
OUT_DIR=$(realpath ~/mmml/WINOGROUND_BUTD)
IMG_DIR=$(realpath ~/mmml/why-winoground-hard/dataset/images)

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm     --mount src=$IMG_DIR,dst=/img,type=bind,readonly     --mount src=$OUT_DIR,dst=/output,type=bind  --mount src=$(realpath ~/mmml),dst=/base,type=bind   -w /src -it chenrocks/butd-caffe:nlvr2     bash
```
Otherwise if you want to generate directly, replace `bash` with
```
bash -c "python tools/generate_npz.py --gpu 0"
```
this will generate save the RoI features as `npz`s in `$OUT_DIR`


Alternatively, there is a [pytorch implementation of Faster-RCNN](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome.git), although this is also outdated as of now since some custom cuda kernels no longer compiles with newer versions of cuda.

This is what works for me for installation for the pytorch version.
```
CUDA_VISIBLE_DEVICES=0
OUT_DIR=$(realpath ~/mmml/WINOGROUND_BUTD)
IMG_DIR=$(realpath ~/mmml/why-winoground-hard/dataset/images)

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm     --mount src=$IMG_DIR,dst=/img,type=bind,readonly     --mount src=$OUT_DIR,dst=/output,type=bind  --mount src=$(realpath ~/mmml),dst=/base,type=bind --mount src=$WORKDIR,dst=/src,type=bind -p 8888:8888  -w /src -it pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime bash
pip install -r requirements.txt
apt-get update
apt-get install g++ -y
cd lib
python setup.py build develop
```
There is no custom generate_npz script though, so the interface might be different with Chen's Caffe version.
