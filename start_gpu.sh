#!/usr/bin/zsh

USER_NAME=docker
NAME=fm_gpu
PORT=9999
sudo docker build --build-arg USER_ID=${UID} \
                  --build-arg USER_NAME=$USER_NAME \
                  --build-arg NAME=$NAME \
                  -f Dockerfile.gpu \
                  -t $NAME .
sudo docker run --gpus all \
                --name $NAME \
                --init \
                -p $PORT:$PORT \
                -v $PWD/:/home/$USER_NAME/$NAME/ \
                -tid $NAME \
                jupyter lab --no-browser --port=$PORT --ip=0.0.0.0

