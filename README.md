# FaceAI
# add by milk
一款优秀的人脸检测、人脸识别、视频识别、文字识别等智能AI项目。

>开发环境：Windows 10（x64） + Python 3
>
>其他库版本：OpenCV 3.4.1、Dlib 19.8.1、face_recognition 1.2.2、Tesseract OCR 4.0.0-beta.1

## 技术实现方案 ##

下面列举一下，相应模块使用到技术

	人脸识别：OpenCV / Dlib

	人脸检测：face_recognition
	
	视频识别：OpenCV + face_recognition

	文字识别：Tesseract OCR


## 教程文档 ##

[OpenCV环境搭建](doc/settingup.md)

[Tesseract OCR文字识别](doc/tesseractOCR.md)

[图片人脸检测（OpenCV版）](doc/detectionOpenCV.md)

[图片人脸检测（Dlib版）](doc/detectionDlib.md)

[视频人脸检测（OpenCV版）](doc/videoOpenCV.md)

[视频人脸检测（Dlib版）](doc/videoDlib.md)

[绘制脸部轮廓](doc/faceRecognitionOutline.md)

[数字化妆](doc/faceRecognitionMakeup.md)

[视频人脸识别](doc/faceRecognition.md)

## 功能预览 ##

**绘制脸部轮廓**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/face_recognition-outline.png)


----------

**人脸68个关键点标识**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/dlib68.png)


----------

**视频人脸检测**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/video-jiance.gif)


----------

**视频人脸识别**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/faceRecognition.gif)


----------

**数字化妆**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/faceRecognitionMakeup.png)

## TODO ##

OpenCV + Tesorflow模型的人脸识别

Dlib模型训练方法

Tesseract模型训练方法

文章最下方，添加“下节”导航

