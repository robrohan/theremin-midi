#!/bin/sh

set -e

pip3 install -r requirements.txt

echo ""                              > log.log

# echo "fetching data..."           >> log.log
# python3 src/prep/0_fetch_data.py         >> log.log

# echo "midi to text..."           >> log.log
# python3 src/v2/prep.py         >> log.log

# # SentencePiece Training
# echo "SP training..."                >> log.log
# python3 src/v2/tokenization_train.py              >> log.log

# GPT training
echo "training..."                >> log.log
python3 src/v2/train.py              >> log.log

# echo "uploading results..."       >> log.log
# python3 src/v2/0_upload_checkpoints.py >> log.log

echo "done."                      >> log.log
