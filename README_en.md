Translation for English Documentations are Working In Progress

# Functions #

1. Face Detection and Recognition（From Image or Video）
2. Facial Landmark
3. Image compositing（e.g. Wear Hat For a Figure)
4. Face Makeup（e.g. Lipstick, Eyebrow, Eyes, etc）
5. Gender Recognition
6. Emotion Recognition（e.g. Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral）
7. Video Object Extraction
8. Image Denoising（e.g. Watermark Removal）
9. Image Colorization
10. Eye Tracking（W.I.P）
11. Face Replacement（W.I.P）

#  Development Environment #

- Windows 10（x64）
- Python 3.6.4
- OpenCV 3.4.1
- Dlib 19.8.1
- face_recognition 1.2.2
- keras 2.1.6
- tensorflow 1.8.0
- Tesseract OCR 4.0.0-beta.1


# Tutorials #

[OpenCV Setup](doc-en/settingup.md)

[Tesseract OCR Text Recognition](doc-en/tesseractOCR.md)

[Face Detection From Image（OpenCV based）](doc-en/detectionOpenCV.md)

[Face Detection From Image（Dlib based）](doc-en/detectionDlib.md)

[Face Detection From Video（OpenCV based）](doc-en/videoOpenCV.md)

[Face Detection From Video（Dlib based）](doc-en/videoDlib.md)

[Face Outline Recognition](doc-en/faceRecognitionOutline.md)

[Face Makeup](doc-en/faceRecognitionMakeup.md)

[Face Recognition From Video](doc-en/faceRecognition.md)

[Image Compositing](doc-en/compose.md)

[Gender Recognition](doc-en/gender.md)

[Emotion Recognition](doc-en/emotion.md)

[Video Object Extraction](doc-en/hsv-opencv.md)

[Image Denoising](doc-en/inpaint.md)


# Other Related Tutorials #

[Ubuntu Software And pip Sources Update](doc-en/ubuntuChange.md)

[pip/pip3 Update To Repository Inside China Mainland——Windows](doc-en/pipChange.md)

[Chinese Font Support In OpenCV](doc-en/chinese.md)

[Mouse Drawing——Based on OpenCV](doc-en/opencv/mouse.md)


# Preview #

**Face Outline Recognition**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/face_recognition-outline.png" width = "250" height = "300" alt="绘制脸部轮廓" />

----------

**68-point Facial Landmark Detection**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/dlib68.png" width = "230" height = "300" alt="人脸68个关键点标识" />

----------

**Image Compositing**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/compose.png" width = "200" height = "300" alt="头像特效合成"  />

----------

**Gender Recognition**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/gender.png" width = "430" height = "220" alt="性别识别"  />

----------

**Emotion Recognition**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/emotion.png" width = "250" height = "300" alt="表情识别"  />

----------

**Face Makeup**

<img src="https://raw.githubusercontent.com/vipstone/faceai/master/res/faceRecognitionMakeup-1.png" width = "450" height = "300" alt="视频人脸识别"  />

----------

**Face Detection From Video**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/video-jiance.gif)

----------

**Face Recognition From Video**

![](https://raw.githubusercontent.com/vipstone/faceai/master/res/faceRecognition.gif)

----------

**Video Object Extraction**

![](http://icdn.apigo.cn/opencv-hsv.gif)

----------

**Image Denoising**

![](http://icdn.apigo.cn/inpaint.png?2)

----------

**Image Colorization**

![](http://icdn.apigo.cn/colorize-faceai.png)

----------

# Background #

Requirements

	Face Recognition ：OpenCV / Dlib / face_recognition
	
	Gender Recognition：keras + tensorflow
	
	Text Recognition：Tesseract OCR


### TODO ###

Face Replacement——W.I.P

Eye Tracking——W.I.P

Dlib Performance Optimization

Dlib Model Training

Tesseract Model Training



# Contributor

[archersmind](https://github.com/archersmind)

[rishab-sharma](https://github.com/rishab-sharma)
