#FROM tensorflow/tensorflow:2.8.0-gpu
FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

# Working directory
WORKDIR /

# Project files
COPY . /

# Update software in Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        vim \
        python3-opencv


# Python packages
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

#/home/app/requirements.txt