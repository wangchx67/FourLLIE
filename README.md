# FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information

## Introduction

This is the official pytorch implementation of "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information" **(ACM MM 2023)**

![pipeline](https://github.com/wangchx67/FourLLIE/blob/main/figs/pipeline.png)

We design a two-stage framework to enhance low-light images with the help of the Fourier frequency information. In the first stage, we improve the lightness of low-light images by estimating the amplitude transform map in the Fourier space. In the second stage, we introduce the Signal-to-Noise-Ratio (SNR) map to provide the prior for integrating the global Fourier frequency and the local spatial information, which recovers image details in the spatial space. With this ingenious design, FourLLIE outperforms the existing state-of-the-art (SOTA) LLIE methods on four representative datasets while maintaining good model efficiency. 

## Installation

```
conda create --name FourLLIE --file requirements.txt
conda activate FourLLIE
```

## Train

You can modify the training configuration (e.g., the path of datasets, learning rate, or model settings) in `./options/train/LOLv2_real.yml` and run:

```
python train.py -opt ./options/train/LOLv2_real.yml
```

## Test

Modify the testing configuration in `./options/test/LOLv2_real.yml` and run:

```
python test.py -opt ./options/test/LOLv2_real.yml
```

## Datasets

- LOL-real and LOL-sys can be found in [here](https://github.com/flyywh/SGM-Low-Light).
- LSRW-Huawei and LSRW-Nikon can be found in [here](https://github.com/JianghaiSCU/R2RNet).

## Pre-trained

We update the pre-trained model in the `./pre-trained`. Note that the initial models of LSRW is missing and we re-trained them. The metric results may not exactly match the results reported in paper. You can freely choose any version.

## Acknowledgement

This repo is based on [SNR-Aware](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).

## Citation Information

If you find the project useful, please cite:

```
@inproceedings{wang2023fourllie,
  title={FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information},
  author={Wang, Chenxi and Wu, Hongjun and Jin, Zhi},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7459--7469},
  year={2023}
}
```

