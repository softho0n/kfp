#!/bin/bash

IMAGE="softhoon/torch-preprocess-image"

docker build -t $IMAGE .
docker push $IMAGE
