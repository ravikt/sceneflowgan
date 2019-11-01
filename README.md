[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ravikt/sceneflowgan/blob/master/LICENSE)

<!---
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/3d-vehicle-tracking.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ucbdrive/3d-vehicle-tracking/context:python)
--->
# SceneFlowGAN

This repository contains Keras implementation of our paper 

A Conditional Adversarial Network for Scene Flow Estimation (RO-MAN 2019)

[Ravi Kumar Thakur](https://ravikt.github.io/) and [Snehasis Mukherjee](https://sites.google.com/a/iiits.in/snehasis-mukherjee/)

## Abstract

<p style="text-align: justify">
The problem of Scene flow estimation in depth videos has been attracting attention of researchers of robot vision, due to its potential application in various areas of robotics. The conventional scene flow methods are difficult to use in reallife applications due to their long computational overhead. We propose a conditional adversarial network SceneFlowGAN for scene flow estimation. The proposed SceneFlowGAN uses loss function at two ends: both generator and descriptor ends. The proposed network is the first attempt to estimate scene flow using generative adversarial networks, and is able to estimate both the optical flow and disparity from the input stereo images simultaneously. The proposed method is experimented on a large RGB-D benchmark sceneflow dataset
</p> 

<img src="misc/SceneFlowGAN.jpg" width="600">

## Requirements

The code has been tested on Ubuntu 16.04 with CUDA 9.0. Python2 and Keras are required. Relevant python libraries can installed (inside virtual environment) using: 
```bash
pip3 install -r requirements.txt
```
Alternatively, a docker image can be created for running SceneFlowGAN inside a container. All the dependencies are included in the Dockerfile. 

## Acknowledgement

We would like to thank [Soumen Ghosh](https://sites.google.com/site/soumenca/) and [Shiv Ram Dubey](https://sites.google.com/site/shivram1987/) for providing feedback and insightful discussion. The work was supported by Department of Science and Technology, Government of India under Project ECR/2016/00652.

## Reference

Please use the following for citation purpose

    @article{thakur2019conditional,
    title={A Conditional Adversarial Network for Scene Flow Estimation},
    author={Thakur, Ravi Kumar and Mukherjee, Snehasis},
    journal={arXiv preprint arXiv:1904.11163},
    year={2019}
    }

## Note

In case of difficulty in running the code, please post your questions by opening issues. To suggest any improvement to make the code more readable or optimized, open a pull request. 

