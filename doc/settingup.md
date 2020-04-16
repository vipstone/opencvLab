# OpenCV环境搭建 #

本文将介绍OpenCV在Python3.x上的实现，分为Window版和Linux版。

## Windows版环境搭建 ##

> 系统环境：windows 10 + python 3.6 + OpenCV 3.4.1


### 一、安装python和pip ###

python的安装之前在[python自学笔记](https://github.com/vipstone/python)的项目中描述了，在这不做重复说明，有需要的朋友，点击查看：[python环境安装](https://github.com/vipstone/python/blob/master/%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md)

### 二、安装numpy模块 ###

根据上文提示，现在我们已经正确安装了python和pip（安装和管理python包的工具），在正式安装OpenCV之前，首先我们要安装numpy模块。
numpy：是一个定义了数值数组和矩阵类型和它们的基本运算的语言扩展，OpenCV引用了numpy模块，所以安装OpenCV之前必须安装numpy。
`pip install numpy`

本文安装python模块使用的是.whl文件安装的。

**whl文件是什么？**

whl是一个python的压缩包，其中包含了py文件以及经过编译的pyd文件。
`pip install wheel`

**whl安装命令**
> pip3 install 存放路径\xxx.whl

回到主题，我们是要安装numpy模块的。

第一步：先去网站下载对应的numpy版本，下载地址：https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy 本人是64为系统python3.6所以对应的最新版本是：numpy‑1.14.2+mkl‑cp36‑cp36m‑win_amd64.whl，点击下载到D:根目录。

百度云链接：https://pan.baidu.com/s/10RefansrC4_0zsNehjyKTg

提取密码：gua3


第2步：启动命令窗体运行
> pip3 install d:\numpy‑1.14.2+mkl‑cp36‑cp36m‑win_amd64.whl

命令窗体显示：

Processing d:\numpy-1.14.2+mkl-cp36-cp36m-win_amd64.whl

Installing collected packages: numpy

Successfully installed numpy-1.14.2+mkl

说明已经安装成功。


### 三、安装OpenCV ###
同样安装OpenCV模块和numpy方式类似。
命令行安装：
`pip install opencv-python`
手动安装：
第1步：首先去网站下载OpenCV对应的.whl版本压缩包，网址：https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv 本人下载的版本是：opencv_python‑3.4.1‑cp36‑cp36m‑win_amd64.whl 64位系统对应python3.6的，下载到d盘根目录。

百度云链接：https://pan.baidu.com/s/10RefansrC4_0zsNehjyKTg

提取密码：gua3

第2步：启动命令窗体运行
> pip3 install d:\opencv_python-3.4.1-cp36-cp36m-win_amd64.whl

窗体显示：

Processing d:\opencv_python-3.4.1-cp36-cp36m-win_amd64.whl

Installing collected packages: opencv-python

Successfully installed opencv-python-3.4.1

说明安装成功。

### 四、运行OpenCV ###
到此，我们的环境配置已经完成了，终于到了可以撸代码的时刻了，想想还有一点小激动呢。


``` python
import cv2

print(cv2.__version__)

# 输出：4.2.0
```
上面我们简单的打印了OpenCV的版本号，如果能正常输出不报错，说明我们已经把OpenCV的python环境搭建ok了。

什么？感觉还不过瘾，那就来撸一张图，用OpenCV把它展示出来，代码如下：
``` python
import cv2

filepath = "img/meinv.png" #这里替换为你电脑照片的路径，注意不要包含中文，斜杠和反斜杠务必转换
img = cv2.imread(filepath)
cv2.namedWindow('Image')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

展示效果如图：![正在加载图片](https://raw.githubusercontent.com/vipstone/opencvLab/master/res/show-meinv.png)


	
----------
## Linux版环境搭建 ##

>Ubuntu 18.04
>Python 3.6.5
>Pip 10.0.1
>Numpy 1.14.3
>OpenCV 3.4.0

Ubuntu有一个好处就是内置Python环境，不需要像Windows那样在为Python环境折腾了，但要注意的是Ubuntu本身自带的apt-get和安装的pip的数据源是国外的，所以使用起来都很慢，一定要把apt-get和pip的数据源更换成国内的，请移步到：[《Ubuntu apt-get和pip源更换》](http://www.cnblogs.com/vipstone/p/9038023.html)

### 正式安装 ###
根据上面的提示，你已经配置好了开发环境，现在需要正式安装了，当然Ubuntu的安装也比Windows简单很多，只需要使用pip安装包，安装相应的模块即可。

#### 安装Numpy ####
使用命令：pip3 install numpy

使用命令：python3，进入python脚本执行环境，输入代码查看numpy是否安装成功，以及numpy的安装版本：
```
import numpy 

numpy.__version__
```
正常输入版本号，证明已经安装成功。

如图：![](http://icdn.apigo.cn/numpy-setup-success.png)

#### 安装OpenCV ####
OpenCV的安装在Ubuntu和numpy相似，使用命令：
>pip3 install opencv-python

使用命令：python3，进入python脚本执行环境，输入代码查看OpenCV版本：
```
import cv2 

cv2.__version__
```
正常输出版本号，证明已经安装成功。

# 常见错误 #

错误一、python3: Relink `/lib/x86_64-linux-gnu/libudev.so.1` with `/lib/x86_64-linux-gnu/librt.so.1` for IFUNC symbol `clock_gettime`
Segmentation fault (core dumped)

解决方案：apt install python3-opencv


----------

