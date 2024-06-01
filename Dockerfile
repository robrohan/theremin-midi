FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
        python3 python3-pip fluidsynth \
    && rm -rf /var/lib/apt/lists/*

COPY ./src ./src
COPY ./output ./output
COPY ./input ./input
COPY requirements.txt ./
COPY train.sh ./train.sh

RUN pip3 install minio
RUN python3 -m pip install -r requirements.txt --upgrade -t .

ENV DEBUG_LEVEL 10

CMD ["./train.sh"]
