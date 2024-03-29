```bash

  _____               ____ _                                 
 |  ___|_ _  ___ ___ / ___| |__   __ _ _ __   __ _  ___ _ __ 
 | |_ / _` |/ __/ _ \ |   | '_ \ / _` | '_ \ / _` |/ _ \ '__|
 |  _| (_| | (_|  __/ |___| | | | (_| | | | | (_| |  __/ |   
 |_|  \__,_|\___\___|\____|_| |_|\__,_|_| |_|\__, |\___|_|   
                                             |___/                         

```


2019 GraduationProjectFinal - SKKU
=======================


                                             
# Author

--------------------------------------

JUNBEOM HEO

Department of Computer Science and Engineering<br>
College of Software <br>
SungKyunKwan University  <br>
2066, Seobu-ro, Jangan-gu, Suwon-si, Gyeonggi-do, Republic of Korea<br>
E-mail: trafalgar23@naver.com / bamy@skku.edu<br>

--------------------------------------

# Face Changer

 * Language: Python
 * Libraries: Opencv, Dlib, PyGame, OpenGL, Numpy, Scipy
 * Principal Techniques
   * Face Recognition
     * frontal face detector of dlib ( dlib.get_frontal_face_detector)
     * facial landmark recognizer
   * Measuring movement of object using camera  (providing Facial recogition guideline)
     * Calculating magnitude of optical flow
   * Composing face to video
     * Mapping to recognized face by dlib to face of model in video 
     * Composing face based on Facial Landmark 

# Environment

MAC OSX <br>
(also possible in Windows, some commands can be different)

# Prerequisites

Anaconda (3.X recommended) 

# Instructions

```shell
$ conda create -n myenv python=3.5 anaconda
```

```shell
$ source activate myenv
```

```shell
$ conda install -c menpo dlib=18.18
```

```shell
$ conda install -c https://conda.binstar.org/menpo opencv
```

```shell
$ pip install PyOpenGL PyOpenGL_accelerate
```

```shell
$ pip install pygame
```

```shell
$ pip install numpy
```

<em>Intel MKL FATAL ERROR: Error on loading function mkl_blas_avx_xdcopy</em><br>
<u>If this error occurs, link below will be help</u><br>

```shell 
$ conda install nomkl numpy scipy scikit-learn numexpr
```

```shell
$ conda remove mkl mkl-service
```

[Error solution reference](https://github.com/pyinstaller/pyinstaller/issues/2175#issuecomment-245438409)


If you successfully ran this code, you will see <br>

![runningImage](./image/success.png)

# Description of Codes

```bash


├── FaceChange.py
├── FaceRendering.py
├── ImageProcessing.py
├── NonLinearLeastSquares.py
├── __pycache__
│   ├── FaceRendering.cpython-35.pyc
│   ├── ImageProcessing.cpython-35.pyc
│   ├── NonLinearLeastSquares.cpython-35.pyc
│   ├── drawing.cpython-35.pyc
│   ├── models.cpython-35.pyc
│   └── utils.cpython-35.pyc
├── drawing.py
├── models.py
└── utils.py

```

* NonLinearLeastSquares.py
  * Gauss Newton Algorithm<br>
   [Link](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm)
 
* drawing.py

* FaceRendering.py

* ImageProcessing.py

* models.py

* utils.py



Reference: https://blog.naver.com/jinohpark79/110189612945

# How it works

* Running Video (2 videos)


[![Video1](http://img.youtube.com/vi/lSGR9kg8rD4/0.jpg)](https://youtu.be/lSGR9kg8rD4?t=0s)<br>
[Running video-1 link](https://youtu.be/lSGR9kg8rD4)<br>

[![Video2](http://img.youtube.com/vi/45nat4zeZWM/0.jpg)](https://youtu.be/45nat4zeZWM?t=0s)<br>
[Running video-2 link](https://youtu.be/45nat4zeZWM)<br>


## More Detail 
[![Video3](http://img.youtube.com/vi/UiSzPO2JShM/0.jpg)](https://youtu.be/UiSzPO2JShM?t=0s)<br>
[Running video-3 link](https://youtu.be/UiSzPO2JShM)<br>

Video3 is explaining case of pressing R
![Press R](./image/pressr.png)

* When you press t, facial points will be displayed.

* When you press r, video will be recorded.

* when you press ESC, program will be executed.


