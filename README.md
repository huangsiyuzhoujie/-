## tensorflow(pc端)    tensorflow-lite(移动端)
#### 安装方法：
> pip install tensorflow-gpu=1.9.0

#### 库地址
[分类]  https://github.com/tensorflow/models/tree/master/research/slim<br>
[目标检测]  https://github.com/tensorflow/models/tree/master/research/object_detection

#### tensorflow-lite  分类过程  
> https://blog.csdn.net/u011092156/article/details/80607601<br>
> https://blog.csdn.net/u011092156/article/details/80642133

#### tensorflow-lite  目标检测过程
一行命令统一修改样本名字 <br>
```bash
ls | cat -n | while read n f; do mv "$f" "Daisy_$n.jpg"; done 
```
转为为tflite文件 部署到手机上的方法<br>
> https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md

tensorflow-lite 目标检测模型的训练过程<br>
> https://github.com/naisy/train_ssd_mobilenet#6

## keras 

## caffe
caffe 部署到移动端使用 ncnn 框架

#### 安装方法
> https://www.linuxidc.com/Linux/2017-01/139313p2.htm<br>
> https://www.cnblogs.com/AbcFly/p/6306201.html

#### caffe  目标检测（移动端）
ssd mobilenet-v1  网络的训练<br>
> https://blog.csdn.net/Chris_zhangrx/article/details/80458515<br>
> https://blog.csdn.net/weixin_39750664/article/details/82502302

ssd mobilenet-v2  网络训练过程<br>
>https://blog.csdn.net/u010986080/article/details/84983310
