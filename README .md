# ELMERF

## Introduction

ELMERF is an open-source semantic segmentation model based on PyTorch.

ELMERF is designed for accurate and robust organ-level phenotyping of rice seedlings under salt stress.
It enables precise segmentation of green leaves, yellow leaves, and roots from RGB images, and supports downstream phenotypic trait extraction for salt-tolerance evaluation and genetic analysis.

The main branch works with PyTorch 1.8+.

---

## Installation

Please refer to the following documentation for detailed instructions:

- [Installation guide](docs/en/get_started.md)
- [Dataset preparation](docs/user_guides/2_dataset_prepare.md)

---

## Get Started_ELMERF

For general usage and design principles, please refer to the official MMSegmentation documentation:

- [MMSegmentation Overview](https://mmsegmentation.readthedocs.io/en/latest/)
- [MMSegmentation User Guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html)
- [Advanced Tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html)

To train the model, run the training script and configure the corresponding config file:

- [tools/train.py](tools/train.py)

To generate segmentation masks from pretrained models, use the inference script:

- [tools/pre_mask.py](tools/pre_mask.py)

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

```
## License

This project is released under the Apache License 2.0.
