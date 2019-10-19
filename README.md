# SceneFlowGAN

This repository contains Keras implementation of our paper 

A Conditional Adversarial Network for Scene Flow Estimation (RO-MAN 2019)

[Ravi Kumar Thakur](https://ravikt.github.io/) and [Snehasis Mukherjee](https://sites.google.com/a/iiits.in/snehasis-mukherjee/)

<img src="misc/SceneFlowGAN.jpg" width="600">

## Requirements

The code has been tested on Ubuntu 16.04 with CUDA 9.0. Python2 and Keras are required. Relevant python libraries can installed (inside virtual environment) using: 
```bash
pip3 install -r requirements.txt
```
Alternatively, a docker image can be created for running SceneFlowGAN inside container. All the dependencies are included in the Dockerfile. 

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

