## tensorflow(pc端)    tensorflow-lite(移动端)
安装方法：
> pip install tensorflow-gpu=1.9.0

[分类]  https://github.com/tensorflow/models/tree/master/research/slim<br>
[目标检测]  https://github.com/tensorflow/models/tree/master/research/object_detection

tensorflow-lite  分类过程  
> https://blog.csdn.net/u011092156/article/details/80607601<br>
> https://blog.csdn.net/u011092156/article/details/80642133

tensorflow-lite  目标检测过程
> `一行命令统一修改样本名字` <br>
```bash
>> ls | cat -n | while read n f; do mv "$f" "Daisy_$n.jpg"; done 
```
