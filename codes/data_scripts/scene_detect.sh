#!/bin/bash

SRC_DIR=/home/xiyang/Downloads/VideoEnhance/train_ref
DST_DIR=/home/xiyang/Downloads/VideoEnhance/train_ref_scene_detect_thres35
mkdir "$DST_DIR"
FILES=$(ls $SRC_DIR | grep .y4m)

for FILE in $FILES
do
    FILENAME="${FILE:0:-4}"
    echo "$FILENAME"
    scenedetect --input $SRC_DIR/$FILE --output $DST_DIR list-scenes detect-content -t 35
done