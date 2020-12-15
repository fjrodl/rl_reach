#!/bin/bash

PARENT="ubuntu:16.04"
PYTORCH_DEPS="cpuonly"
TAG="rlreach-cpu"
VERSION="latest"

# docker build --build-arg PARENT_IMAGE=ubuntu:16.04 --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t rlreach-cpu:latest .

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .