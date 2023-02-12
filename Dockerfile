ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    wget\
    bash\
    openssh-server &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN apt-get update && apt-get install -y --no-install-recommends
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3.10 -y
RUN apt-get install python3-pip -y


COPY requirements.txt /opt/ckpt/requirements.txt
WORKDIR /opt/ckpt
RUN pip3 install -r requirements.txt
RUN pip3 install triton==2.0.0.dev20221120
RUN pip3 install https://github.com/runpod/runpod-python/archive/main.zip

COPY . /opt/ckpt

RUN python3 model_fetcher.py

CMD [ "python3", "-u", "/opt/ckpt/runpod_infer.py"]
