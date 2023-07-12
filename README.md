# SAM-U 
## MICCAI2023 UNSURE workshop submission
* This repository provides the code for our accepted MICCAI'2023 UNSURE workshop paper "SAM-U: Multi-box prompts triggered uncertainty estimation for reliable SAM in medical image"
* Official implementation [SAM-U](https://arxiv.org/pdf/2307.04973.pdf)

## Introduction
Recently, Segmenting Anything Model has taken a significant step towards general artificial intelligence. Simultaneously, its reliability and fairness have garnered significant attention, particularly in the field of healthcare. In this study, we propose a multi-box prompt triggered uncertainty estimation for SAM cues to demonstrate the reliability of segmented lesions or tissues. We estimate the distribution of SAM predictions using Monte Carlo with prior distribution parameters, employing different prompts as a formulation of test-time augmentation. Our experimental results demonstrate that multi-box prompts augmentation enhances SAM performance and provides uncertainty for each pixel. This presents a groundbreaking paradigm for a reliable SAM.

![pdf](https://github.com/rabbitdeng/SAM-U/blob/main/Framework.pdf)
