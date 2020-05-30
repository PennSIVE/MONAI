#!/bin/bash

# remember to add connection to acl with xhost, get display IP with `ifconfig en0 | grep 'inet '`
docker run --rm -ti --name MONAI -e QT_X11_NO_MITSHM="1" \
    -e DISPLAY="192.168.0.148:0" \
    -v /Users/tim/Downloads/Task09_Spleen:/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    pennsive/monai:latest python /src/main.py
