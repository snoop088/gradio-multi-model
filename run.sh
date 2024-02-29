#!/bin/bash

# This is your Docker run command
docker run --name gradio-model-app --rm --gpus all -p 7870:7860 \
--mount type=bind,source=/home/snoop/Documents/docker/gradio-multi-model/app,target=/code/app \
--mount type=bind,source=/mnt/models/Models,target=/code/models \
gradio-models-app:latest
