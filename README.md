## GraduationProjectFinal

# Face Changer


* How to Implement?

# Environment

MAC OSX

# Prerequisites

Anaconda

# Instructions

```cmd
conda create -n myenv python=3.5 anaconda
```

```cmd
source activate myenv
```

```cmd
conda install -c menpo dlib=18.18
```

```cmd
conda install -c https://conda.binstar.org/menpo opencv
```

```cmd
pip install PyOpenGL PyOpenGL_accelerate
```

```cmd
pip install pygame
```

```cmd
pip install numpy
```



<em>Intel MKL FATAL ERROR: Error on loading function mkl_blas_avx_xdcopy</em>
If this error occurs, link below will be help<br>

```cmd 
conda install nomkl numpy scipy scikit-learn numexpr
```

```cmd
conda remove mkl mkl-service
```
[Error solution reference](https://github.com/pyinstaller/pyinstaller/issues/2175#issuecomment-245438409)



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

# Related
<ol>
<li>
https://github.com/MarekKowalski/FaceSwap
 </li>
 <li>
https://github.com/hrastnik/FaceSwap
 </li>
 </ol>



