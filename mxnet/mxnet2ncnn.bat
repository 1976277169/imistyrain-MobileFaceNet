@echo off

set tools="D:/CNN/ncnn/build/tools/mxnet/Release/mxnet2ncnn.exe"
echo "converting to ncnn"
%tools% model-symbol.json model-0000.params ../models/MobileFaceNet.param ../models/MobileFaceNet.bin
echo "Done"
pause