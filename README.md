# TUNI:
The experimental code of paper "TUNI: Textual Unimodal Detector for Identity Inference in CLIP Models".

## Introduction
This anonymous repository contains the code that implements the TUNI described in our paper "TUNI: Textual Unimodal Detector for Identity Inference in CLIP Models" . This work studies the identity inference in CLIP models with only text data of an individual, a process of determining whether the PII of the indicidual is in the target model training process or not.

## Requirements
Our code is implemented and tested on Pytorch with the other packages in requirements.txt(some conducted in the local CPU environment and some remote on GPUs), you can quickly establish your environment by anaconda through: `conda create --name 'TUNI' --file requirements.txt` `conda activate 'TUNI'`

## Dataset and Pretrained Models
Please download these dataset and pretrained models from public sources and put them in place.
* The target CLIP models can be accessed from following link: https://github.com/D0miH/does-clip-know-my-face/releases
* The dataset to evaluate our detector(containing names of individuals both in and out of the training dataset of the target CLIP models) can be accessed from following link: https://github.com/D0miH/does-clip-know-my-face/tree/main/cc3m_experiments/conceptual_captions_facescrub_member_info
* The dataset to choose from for the initial image in the optimization process can be accessed from following link: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or 
https://susanqq.github.io/UTKFace/ (any public dataset with images of individuals can be used)
* The face feature extraction model DeepFace can be accessed from following link: https://github.com/jagtapraj123/Knowledge_Distillation-Face_Recognition




