# Misalignment Data Augmentation
The data augmentation is proposed in [Nature Scientific Reports 2023](https://www.nature.com/articles/s41598-023-46747-z).

If you use this augmentation please cite the following work:
```
Kovacs, Balint, et al.
"Addressing image misalignments in multi-parametric prostate MRI
for enhanced computer-aided diagnosis of prostate cancer."
Scientific Reports 13.1 (2023): 19805.
```

## Usage
The nnU-Net trainer `nnUNetTrainer_Misalign.py` generates possible misalignments/registration errors
between the input image modalities/channels during training to teach the network to become robust
to them thereby increasing their performance. The extension is simple, we just appended the
`tr_transforms` with the `MisalignTransform` transformation with the following parameters:
   - `im_channels_2_misalign`: on which image channels should the transformation be applied
   - `label_channels_2_misalign`: on which segmentation channels should the transformation be applied
   - `do_squeeze`: whether misalignment resulted from squeezing is necessary
   - `sq_x`, `sq_y`, `sq_z`: probability of the transformation per organs
   - `p_sq_per_sample`: probability of the transformation per sample
   - `p_sq_per_dir`: probability of the transformation per direction
   - `do_rotation`: whether misalignment resulted from rotation is necessary
   - `angle_x`, `angle_y`, `angle_z`: rotation angels per axes, randomly sampled from interval.
   - `p_rot_per_sample`: probability of the transformation per sample
   - `p_rot_per_axis`: probability of the transformation per axes
   - `do_transl`: whether misalignment resulted from rotation is necessary
   - `tr_x`, `tr_y`, `tr_z`: shift/translation per directions, randomly sampled from interval.
   - `p_transl_per_sample`: probability of the transformation per sample
   - `p_transl_per_dir`: probability of the transformation per direction

**Important suggestions for its successful utilization:**

The ability of the method to cope with misalignments is limited depending on the initial overlap between
the image modalities, and the type and amplitude of the augmentations. Therefore, its parameters have to be
adapted to the exact application, consider the following aspects:
* An initial registration might still be needed.
* There might be misalignments from other sources in your dataset, you can extend the augmentation scheme
with them.
* Choose the probability and the amplitude of the augmentation wisely.

Misalignment augmentation is originally proposed to enhance diagnostic performance, but you can use it
for other multi-modal/multi-channel applications.

## Installation
The repository is forked from nnU-Net v2, but contains no pretrained network so it can be used as an integrative
framework for model development
1) Create a new conda environment with the recent version of Python (nnU-Net v2 supports 3.9 or newer version),
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