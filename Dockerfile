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

RUN pip install --upgrade pip
RUN pip install minio
RUN pip install --upgrade -r requirements.txt

ENV DEBUG_LEVEL 10

CMD ["./train.sh"]
