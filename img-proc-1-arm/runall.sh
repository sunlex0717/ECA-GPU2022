#!/bin/bash
for f in `cat test-images/001-bitmap-list.txt` ; do
    echo ----- $f
    ./arm_img_proc test-images/$f ./$f
done
