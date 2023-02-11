#!/bin/bash
IMAGE="softhoon/torch-train-image"

docker build -t $IMAGE .
docker push $IMAGE
