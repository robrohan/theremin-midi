#!/bin/sh

set -e

pip3 install -r requirements.txt

echo ""                              > log.log

echo "fetching data..."           >> log.log
python3 src/prep/0_fetch_data.py         >> log.log

echo "training..."                >> log.log
python3 src/v1/train.py              >> log.log

echo "uploading results..."       >> log.log
python3 src/v1/0_upload_checkpoints.py >> log.log

echo "done."                      >> log.log
