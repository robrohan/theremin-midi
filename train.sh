#!/bin/sh
################################
# This file is the main kick off point within the docker container.
# Note that if the .pt file exists from the download from minio then
# this version is already trained, and just exit. This will cause a k8
# crash and eventually have a crash back loop and stop training.
#
# This is better done as a cronjob.
#
set -ex

pip3 install -r requirements.txt

echo ""                              > log.log

mkdir -p ./models/

echo "fetching data..."           >> log.log
python3 src/v2/0_fetch_data.py         >> log.log

# Done before hand
# echo "midi to text..."           >> log.log
# python3 src/v2/prep.py         >> log.log

# # SentencePiece Training
# echo "SP training..."                >> log.log
# python3 src/v2/tokenization_train.py              >> log.log

if [ -f "./models/$VERSION/theremin.pt" ]; then
    echo "Already trained!"
    exit 1
fi

# GPT training
echo "training..."                >> log.log
python3 src/v2/train.py              >> log.log

echo "uploading results..."       >> log.log
python3 src/v2/0_upload_checkpoints.py >> log.log

echo "done."                      >> log.log
