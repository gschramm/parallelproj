#FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# unminimize image
RUN yes | unminimize

# install cmake
RUN 1 | apt install -y cmake

# install git
RUN apt install -y git

# clone parallelproj
RUN git clone https://github.com/gschramm/parallelproj.git

# build and install parallelproj
RUN mkdir build && cd build && cmake ../parallelproj && cmake --build . && cmake --install .
