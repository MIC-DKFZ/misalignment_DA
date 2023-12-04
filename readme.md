# Misalignment Data Augmentation
The data augmentation is proposed in [Nature Scientific Reports 2023](https://www.nature.com/articles/s41598-023-46747-z).

If you use this augmentation please cite the following work:
```
Kovacs, Balint, et al.
"Addressing image misalignments in multi-parametric prostate MRI
for enhanced computer-aided diagnosis of prostate cancer."
Scientific Reports 13.1 (2023): 19805.
```

## Installation
The repository is forked from nnU-Net v2, but contains no pretrained network so it can be used as an integrative
framework for model developement
1) Create a new conda new conda environment with the recent version of Python (nnU-Net v2 supports 3.9 or newer version),
as an example: `conda create --name nnUNet_AnatInf python=3.9`
2) Install pytorch with the most recent CUDA version by following the instructions on the
[PyTorch Website](https://pytorch.org/get-started/locally/).
3) Clone this repository and install its dependencies:
```
https://github.com/MIC-DKFZ/anatomy_informed_DA.git
cd anatomy-informed-da
pip install -e .
```
You can find more information in the
[nnU-Net installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).


## Copyright
Copyright German Cancer Research Center (DKFZ) and contributors.
Please make sure that your usage of this code is in compliance with its
[license](https://github.com/MIC-DKFZ/anatomy_informed_DA/blob/master/LICENSE).

<img src="documentation/assets/dkfz_logo.png" height="100px" />