## 各种issue
1. 运行 caffe ssd mobilenet-v2 中的 load_caffe_weights.py 时出现cudnn的问题，取消 deploy.prototxt 中的 engine: caffe 的注释<br>
2. cuda9.0 在安装 opencv3 的时候会遇到 CUDA_nppi_LIBRARY 的问题，解决方法如下：<br>
    https://blog.csdn.net/u014613745/article/details/78310916<br>
    “cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CUDA_GENERATION=Kepler ..”<br>
3. 在使用下面的教程中 ssd mobilenet-v2 的时候，会出现 cudnn 的问题，需要将 deploy.prototxt 中的 #engine:caffe 注释掉。<br>
4. 在 python 中使用 cv2.imread 读取为空时表示没有opencv的python接口，安装 opencv-python 即可。<br>
5. 使用 cmake 时出现 libEGL.so 的错误，解决方法:<br>
    https://www.jianshu.com/p/74e9c8697372
-----------------------------------------------------------------------------------------------------------------------

## tensorflow(pc端)    tensorflow-lite(移动端)
#### 1.安装方法：
> pip install tensorflow-gpu=1.9.0

#### 2.数据集处理  tfrecord
先将xml文件转化为csv，然后转化为tfrecord

#### 3.库地址
[分类]  https://github.com/tensorflow/models/tree/master/research/slim<br>
[目标检测]  https://github.com/tensorflow/models/tree/master/research/object_detection

#### 4.tensorflow-lite  分类过程  
> https://blog.csdn.net/u011092156/article/details/80607601<br>
> https://blog.csdn.net/u011092156/article/details/80642133

#### 5.tensorflow-lite  目标检测过程
一行命令统一修改样本名字 <br>
```bash
ls | cat -n | while read n f; do mv "$f" "Daisy_$n.jpg"; done 
```
转为为tflite文件 部署到手机上的方法<br>
> https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md

tensorflow-lite 目标检测模型的训练过程<br>
> https://github.com/naisy/train_ssd_mobilenet#6

#### 6.树莓派上使用 ssdlite-mobilenetv2
>https://blog.csdn.net/xyc2690/article/details/80769899


## keras 

---------------------------------------------------------------------------------------------------------------------
## caffe
caffe 部署到移动端使用 ncnn 框架

#### 1.安装方法
> https://www.linuxidc.com/Linux/2017-01/139313p2.htm<br>
> https://www.cnblogs.com/AbcFly/p/6306201.html<br>
[官方]https://github.com/BVLC/caffe/wiki/OpenCV-3.3-Installation-Guide-on-Ubuntu-16.04

#### 2.数据集处理
分类 转化为 lmdb 文件
> http://www.cnblogs.com/luludeboke/p/7813060.html

#### 3.caffe  目标检测（移动端）
ssd mobilenet-v1  网络的训练<br>
> https://blog.csdn.net/Chris_zhangrx/article/details/80458515<br>
> https://blog.csdn.net/weixin_39750664/article/details/82502302

ssd mobilenet-v2  网络训练过程<br>
>https://blog.csdn.net/u010986080/article/details/84983310

#### 4.caffe 画loss和accuracy曲线
> http://www.mamicode.com/info-detail-2311403.html<br>
> https://blog.csdn.net/auto1993/article/details/71293678<br>
> https://www.cnblogs.com/txg198955/p/6185787.html<br>
> https://blog.csdn.net/u014653401/article/details/59110465


## pytorch
#### pytorch 数据集处理过程
对于分类和目标检测问题<br>
实现数据集的处理类，继承于 torch.util.data.Dataset, 必须实现 __getitem__ __len__ 两个函数。<br>
使用 DataLoader 批量处理样本 dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)。<br>
>> 分类例子
```python
class CaiDataset(Dataset):
    def __init__(self, datadir, labelfile, transforms):
        super(CaiDataset, self).__init__()
        imgs = os.listdir(datadir)
        self.imgs = [os.path.join(datadir, img) for img in imgs]
        self.transforms = transforms
        self.labelfile = labelfile

    def __getitem__(self, index):
        img_path = self.imgs[index]

        labels = open(self.labelfile).readlines()
        labels = [line.rstrip() for line in labels]
        for label in labels:
            label = eval(label)
            for key, value in label.items():
                if key in img_path.split('/')[-1]:
                    self.label = value

        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        label = self.label

        return img, label
    
    def __len__(self):
        return len(self.imgs)
```
>> 目标检测的数据集处理  可参数此处 (pytorch 实现的 各种不同的 ssd 网络) <br>
https://github.com/ShuangXieIrene/ssds.pytorch<br>
>> pytorch中自定义数据集的处理方法<br> 
https://www.pytorchtutorial.com/pytorch-custom-dataset-examples/

----------------------------------------------------------------------------------------------------------------------

## yolov3
#### yolov3 修改源码 批量测试样本并保存在文件夹上
> https://blog.csdn.net/mieleizhi0522/article/details/79989754

#### yolov3 计算map值和recall召回率
> https://blog.csdn.net/cgt19910923/article/details/80524173

voc_eval.py
> https://blog.csdn.net/shawncheer/article/details/78317711

#### yolov3 的可视化过程
> https://blog.csdn.net/shangpapa3/article/details/76687191<br>
> https://blog.csdn.net/vvyuervv/article/details/72868749
