# 视频对象提取 #

与其说是视频对象提取，不如说是视频颜色提取，因为其本质还是使用了OpenCV的HSV颜色物体检测。

# HSV介绍 #
HSV分别代表，色调（H：hue），饱和度（S：saturation），亮度（V：value），由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)；

色调（H：hue）：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°，品红为300°；（OpenCV中H的取值范围为0~180，8bit存储时）；

饱和度（S：saturation）：取值范围为0~255，值越大，颜色越饱和；

亮度（V：value）：取值范围为0(黑色)～255(白色)；

![](http://www.nmt.edu/tcc/help/pubs/colortheory/img/cone.png)

# 效果展示 #
![](http://icdn.apigo.cn/hsv.gif)

# 实现思路 #
如上效果图所示，我们要做的就是把视频中的绿色的小猪佩奇识别出来即可，下面是的识别步骤：

1. 使用PS取的小猪佩奇颜色的HSB值，相当于OpenCV的HSV，不过PS的HSV（HSB）取值是：0~360、0~1、0~1，而OpenCV的HSV是：0~180、0~255、0~255，所以要对ps的hsv进行处理，H/2、SV*255；
1. 使用OpenCV位“与运算”提取HSV的颜色部分画面；
2. 使用高斯模糊优化图片；
3. 图片展示；

PS中工具栏右侧HSB显示：
![](http://icdn.apigo.cn/hsb.png)

# 完整代码 #
```
#coding=utf-8
#HSV转换（颜色提取）

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #在PS里用取色器的HSV
    psHSV = [112, 89, 52]
    diff = 40  #上下浮动值
    #因为PS的HSV（HSB）取值是：0~360、0~1、0~1，而OpenCV的HSV是：0~180、0~255、0~255，所以要对ps的hsv进行处理，H/2、SV*255
    lowerHSV = [(psHSV[0] - diff) / 2, (psHSV[1] - diff) * 255 / 100,
                (psHSV[2] - diff) * 255 / 100]
    upperHSV = [(psHSV[0] + diff) / 2, (psHSV[1] + diff) * 255 / 100,
                (psHSV[2] + diff) * 255 / 100]

    mask = cv2.inRange(hsv, np.array(lowerHSV), np.array(upperHSV))

    #使用位“与运算”提取颜色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #使用高斯模式优化图片
    res = cv2.GaussianBlur(res, (5, 5), 1)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```


