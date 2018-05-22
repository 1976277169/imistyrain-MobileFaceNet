@echo off

set tools="D:/CNN/ncnn/build/tools/mxnet/Release/mxnet2ncnn.exe"
echo "converting to ncnn"
%tools% model-symbol.json model-0000.params ../ncnn/MobileFaceNet.param ../ncnn/MobileFaceNet.bin
echo "Done"
pause