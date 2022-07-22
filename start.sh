#!/bin/sh
arecord -Dac108 -f S32_LE -r 44100 -c 1 record.wav &
python3 main.py
