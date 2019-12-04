# Face_Recognition
基于CNN以及于仕琪人脸检测算法的人脸识别

### 一、编译环境

1. ##### 编译环境：

   python 3.6 + tensorflow 1.14 

2. ##### 于仕琪人脸检测：

   由于使用python接口的于仕琪人脸检测，在测试时，请将作业文件夹中的 "fd-shiqiyu_v2.dll" 文件拷贝到 Python 安装位置的 “DLLs” 子目录下，以便调用于仕琪人脸检测算法。

3. ##### 文件夹文件描述：

   - CNN_Face_Recognition\Faces_Last-1.output\log	     中为训练后的模型文件（太大了不传了，要用的自己训练吧）
   - CNN_Face_Recognition\Faces_Last-1.output\tf_dir     中为Tensorboard可视化文件
   - ‪CNN_Face_Recognition\Face_Recongnize_Train.py      为训练代码
   - CNN_Face_Recognition\Face_Recongnize.py                 为最终结果呈现的代码

4. ##### 运行方式：

   - 将作业文件夹中的 "fd-shiqiyu_v2.dll" 文件拷贝到 Python 安装位置的 “DLLs” 子目录下
   - import必要的包后运行CNN_Face_Recognition\Face_Recongnize.py

5. ##### 数据集：

   数据集使用的是以学号文件夹命名的128*128的人脸数据
   
6. ##### 注：

    这里的于仕琪人脸检测python的接口是这位大佬做的，放下他的github链接： https://github.com/armstrong1972/pySample-for-ShiqiYu
