@echo off

set TOOLS="../tools/Release/caffe2ncnn_mtcnn"

%TOOLS% det1.prototxt det1.caffemodel det1.param det1.bin

%TOOLS% det2.prototxt det2.caffemodel det2.param det2.bin

%TOOLS% det3.prototxt det3.caffemodel det3.param det3.bin

pause
