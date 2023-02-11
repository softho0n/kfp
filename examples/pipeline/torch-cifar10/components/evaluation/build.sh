#!/bin/bash

IMAGE="softhoon/torch-evaluation-image"

docker build -t $IMAGE .
docker push $IMAGE
