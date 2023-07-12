# SAM-U 
## MICCAI2023 UNSURE workshop submission
* This repository provides the code for our accepted MICCAI'2023 UNSURE workshop paper "SAM-U: Multi-box prompts triggered uncertainty estimation for reliable SAM in medical image"
* Official implementation [SAM-U](https://arxiv.org/pdf/2307.04973.pdf)

## Introduction
Recently, Segmenting Anything Model has taken a significant step towards general artificial intelligence. Simultaneously, its reliability and fairness have garnered significant attention, particularly in the field of healthcare. In this study, we propose a multi-box prompttriggered uncertainty estimation for SAM cues to demonstrate the reliability of segmented lesions or tissues. We estimate the distribution of SAM predictions using Monte Carlo with prior distribution parameters, employing different prompts as a formulation of test-time augmentation. Our experimental results demonstrate that multi-box prompts augmentation enhances SAM performance and provides uncertainty for each pixel. This presents a groundbreaking paradigm for a reliable SAM.


<p float="left">
  <img src="sam-u.png?raw=true" width="80%" />

</p>
