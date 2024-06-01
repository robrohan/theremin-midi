#!/bin/sh

set -e

pip install -r requirements.txt

echo "fetching data..."           >> log.log
python src/0_fetch_data.py         >> log.log

echo "training..."                >> log.log
python src/train.py              >> log.log

echo "uploading results..."       >> log.log
python src/0_upload_checkpoints.py >> log.log

echo "done."                      >> log.log
