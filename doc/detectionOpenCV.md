# 图片人脸检测

人脸检测使用到的技术是OpenCV，上一节已经介绍了OpenCV的环境安装，[点击查看](https://github.com/vipstone/faceai/blob/master/doc/settingup.md).

## 功能展示 ##
识别一种图上的所有人的脸，并且标出人脸的位置，画出人眼以及嘴的位置，展示效果图如下：
![](https://raw.githubusercontent.com/vipstone/faceai/master/res/jiance.png)

多张脸识别效果图：
![](https://raw.githubusercontent.com/vipstone/faceai/master/res/jiance-2.png)

## 技术实现思路 ##

图片转换成灰色（降低为一维的灰度，减低计算强度）

图片上画矩形

使用训练分类器查找人脸

## 具体实现代码 ##


**图片转换成灰色**

使用OpenCV的cvtColor()转换图片颜色，代码如下：
```
import cv2

filepath = "img/xingye-1.png"
img = cv2.imread(filepath)
# 转换灰色
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 显示图像
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**图片上画矩形**

使用OpenCV的rectangle()绘制矩形，代码如下：

``` 
import cv2

filepath = "img/xingye-1.png"
img = cv2.imread(filepath)  # 读取图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
x = y = 10  # 坐标
w = 100  # 矩形大小（宽、高）
color = (0, 0, 255)  # 定义绘制颜色
cv2.rectangle(img, (x, y), (x + w, y + w), color, 1)  # 绘制矩形
cv2.imshow("Image", img)  # 显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()  # 释放所有的窗体资源
```


**使用训练分类器查找人脸**

在使用OpenCV的人脸检测之前，需要一个人脸训练模型，格式是xml的，我们这里使用OpenCV提供好的人脸分类模型xml，下载地址：[https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) 可全部下载到本地，本人存放的路径是：C:\Python36\Lib\site-packages\opencv-master\data\haarcascades.

完整实现代码：
```
import cv2

filepath = "img/xingye-1.png"
img = cv2.imread(filepath)  # 读取图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色

# OpenCV人脸识别分类器
classifier = cv2.CascadeClassifier(
    "C:\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
)
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faceRects = classifier.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects):  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        # 框出人脸
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
        # 左眼
        cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        #右眼
        cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                   color)
        #嘴巴
        cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                      (x + 5 * w // 8, y + 7 * h // 8), color)

cv2.imshow("image", img)  # 显示图像
c = cv2.waitKey(10)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
分类器classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))**参数说明**：

gray：转换的灰图

scaleFactor：图像缩放比例，可理解为相机的X倍镜

minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏

minSize：特征检测点的最小尺寸



