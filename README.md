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

## pytorch
#### pytorch 数据集处理过程
对于分类和目标检测问题<br>
>> 实现数据集的处理类，继承于 torch.util.data.Dataset, 必须实现 __getitem__ __len__ 两个函数
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
