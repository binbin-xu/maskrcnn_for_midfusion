#!/bin/bash

demo_dir=$PWD/demo

echo $demo_dir

cd $demo_dir
python3 video_mask_rcnn.py $1