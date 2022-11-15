# EE104 Lab7
## Nasco Arseniev and Stephany Del Greco

# Video Demo: https://youtu.be/A__37ZIEdVk

There are two portions to this lab:

- Part 1: Balloon Flight Game
- Part 2: CNN for Image Recognition

# Part 1: Balloon Flight Game
## Usage
Download zip folder and extract files:
- images folder
- high-scores.txt
- balloonflight.py

Install dependencies:

```bash
$ pip install pgzrun
$ pip install pgzero
$ pip install random
```

Then, run it in Spyder3, or with no arguments:

```bash
$ python balloonflight.py
```

# Part 2: CNN - Image Recognition
## Usage
Run cnn.py in Google Colab (https://colab.research.google.com/)
- Ensure that trained model "Group14_CIFARmodel.h5" is saved and downloaded after running on Colab

Install dependencies:

```bash
$ pip install keras
$ pip install tensorflow
$ pip install ssl
$ pip install PIL
```

Insert challenge test image URL in line 29 of recognizeimages.py and save file
```bash
URL = "https://static.toiimg.com/thumb/msid-67586673,width-1070,height-580,overlay-toi_sw,pt-32,y_pad-40,resizemode-75,imgsize-3918697/67586673.jpg"
```

Then, run it in Spyder3, or with no arguments:

```bash
$ python recognizeimages.py
```
