#!/bin/bash

PARENT="nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04"
PYTORCH_DEPS="cudatoolkit=10.1"
TAG="rlreach-gpu"
VERSION="latest"

# docker build --build-arg PARENT_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04 --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t rlreach-gpu:latest .

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .