# CAM(Class Activation Map)

This repository is an pytorch implementation of [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150). 

<img src="https://github.com/facebookresearch/FixRes/blob/master/image/image2.png" height="190">

It is modified from https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py.

All images are gathered from https://www.kaggle.com/c/dogs-vs-cats.

## Requirements

* python 3.6
* torch 1.2
* torchvision 0.5
* numpy
* matpotlib 

## Usage

To get CAM of images, you have to use a model which consist of [avg-pool, fc] as last 2-layers.  

```python
from CAM import CAM
cam, pre = CAM(model, images,
               last_conv_name='inception5b', fc_name='fc', 
               label=None, normalize=True, resize=True)
```