#!/bin/sh
set -ex

HOST=$1
PROMPT=$2
OUTPUT=$3

curl -X POST "$HOST" \
    -H "Content-Type: audio/midi" \
    --data-binary "$PROMPT" \
    > "$OUTPUT"

