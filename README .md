## Introduction

ELMERF is an open-source semantic segmentation model based on PyTorch.

ELMERF is designed for accurate and robust organ-level phenotyping of rice seedlings under salt stress. 
It enables precise segmentation of green leaves, yellow leaves, and roots from RGB images, and supports downstream phenotypic trait extraction for salt-tolerance evaluation and genetic analysis.

The main branch works with PyTorch 1.8+.

---

## Installation

Please refer to `docs/get_started.md` for installation instructions and 
`docs/dataset_prepare.md` for dataset preparation.

---

## Get Started - ELMERF

Please see the Overview for the general introduction of MMSegmentation.

Please see user guides for the basic usage of MMSegmentation.

There are also advanced tutorials for in-depth understanding of MMSegmentation design and implementation.

Run the `tools/train.py` file and fill in the configuration file to train the model.

A script for generating segmentation masks from pretrained models is provided in `tools/pre_mask.py`.

---

## Citation

If you find this project useful in your research, please cite the following paper:

```bibtex
@article{ELMERF2025,
  title   = {ELMERF: A Deep Learning-Based Image Phenotyping Framework for Salt-Tolerance Evaluation and Genetic Mapping in Rice Seedlings},
  author  = {Li, Yunluo and Meng, Xianghui and Xu, Qiyun and Xu, Ran and Bai, Xiaodong and Yang, Zhuang and Liu, Hongyan},
  journal = {Plant Phenomics},
  note    = {under review},
  year    = {2025}
}
