#!/usr/bin/env bash
#
# record.sh <output.wav>
#
# Records from the Pulse default source into the given WAV file.
# Exits cleanly when it receives SIGINT or SIGTERM.

set -u

OUT="${1:?usage: record.sh <output.wav>}"

# Ensure clean exit on signals.
trap 'exit 0' INT TERM

arecord \
    -q \
    -D pulse \
    -f S16_LE \
    -r 16000 \
    -c 1 \
    "$OUT"

# arecord exits after signal — file is finalized by then.
exit 0
