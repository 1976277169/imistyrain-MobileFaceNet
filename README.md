# MobileFaceNet 一站式解决方法

这是论文[MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)caffe版的复现

## 使用

### Windows

双击打开.sln文件即可

### Linux

* 下载并编译[ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build)
* 编译本项目
```
https://github.com/imistyrain/MobileFaceNet
git clone
cd MobileFaceNet
mkdir build
cd build
cmake ..
make
```

## 配置:

1.拉取并编译[spheraface](https://github.com/wy1iu/sphereface)

2.准备训练环境
将本工程caffe文件夹下的代码放置在sphereface/train目录下
![](https://i.imgur.com/4wml7xQ.jpg)

3.开始训练
```
cd $SPHERFACE_DIR/train/
sh mobilefacenet/mobilefacenet_train.sh

```

## 预训练好的模型：
![](https://i.imgur.com/LqMa5EU.jpg)

## 参考：

* [mobilefacenet-caffe](https://github.com/KaleidoZhouYN/mobilefacenet-caffe)

* [amsoftmax](https://github.com/happynear/AMSoftmax)

* [yonghenglh6/DepthwiseConvolution](https://github.com/yonghenglh6/DepthwiseConvolution)

* [shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)

* [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn)