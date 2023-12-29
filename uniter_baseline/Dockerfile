FROM nvcr.io/nvidia/caffe:19.08-py2

RUN bash -c "pip install easydict"
# RUN bash -c "mkdir -p /src/data/faster_rcnn_models/ && cd /src/data/faster_rcnn_models && wget https://www.dropbox.com/s/tr24q7h0zm2wnjv/resnet101_faster_rcnn_final.caffemodel"

# build cython
COPY ./lib /src/lib
RUN bash -c "cd /src/lib && make"

# build caffe
COPY ./caffe /src/caffe
RUN bash -c "cd /src/caffe && make -j$(nproc) && make pycaffe"

# copy remaining source code
COPY ./ /src/
