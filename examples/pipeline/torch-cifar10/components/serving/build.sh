#!/bin/bash

IMAGE="softhoon/torch-serving-image"

docker build -t $IMAGE .
docker push $IMAGE
