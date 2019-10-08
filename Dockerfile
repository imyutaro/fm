FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

ENV USER_NAME=user
RUN useradd -m $USER_NAME

WORKDIR /home/$USER_NAME
COPY ./req.txt ./
RUN pip install -r ./req.txt

USER $USER_NAME

