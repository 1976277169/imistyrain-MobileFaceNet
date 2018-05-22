#!/bin/bash
# Usage:
# ./code/sphereface_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3
SPHERE_DIR=../tools/caffe-sphereface
GPU_ID=0
$SPHERE_DIR/build/tools/caffe train -solver caffe/mobilefacenet_solver.prototxt -gpu ${GPU_ID} 2>&1 | tee result/sphereface_train.log