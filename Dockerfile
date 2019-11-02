FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i en_US en_US.UTF-8
RUN apt-get -y install zsh
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ JST-9
ENV TERM xterm

# need for jupyter lab
RUN apt-get install -y nodejs npm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

ARG USER_NAME
ARG USER_ID
RUN useradd -u $USER_ID -m $USER_NAME

WORKDIR /home/$USER_NAME
COPY ./req.txt ./
RUN pip install -r ./req.txt

RUN jupyter labextension install jupyterlab_vim

USER $USER_NAME

