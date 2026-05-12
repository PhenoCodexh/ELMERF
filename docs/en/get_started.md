# Installation and data preparation

This project has been tested with the following environment:

- Python 3.8
- PyTorch 1.13.0 + CUDA 11.7
- TorchVision 0.14.0 + CUDA 11.7
- Torchaudio 0.13.0
- MMSegmentation 1.2.2
- MMCV 2.0.0
- MMEngine 0.10.7

## Step 1: Create a new environment

```shell
conda create -n ELMERF python=3.8 -y
conda activate ELMERF
```

## Step 2: Install PyTorch

```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Step 3: Install OpenMMLab dependencies

```shell
pip install mmengine==0.10.7

pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

pip install mmsegmentation==1.2.2
```

## Step 4: Download and install ELMERF

```shell
git clone https://github.com/PhenoCodexh/ELMERF.git
cd ELMERF
pip install -v -e .
```

The `-e` option installs the local source code in editable mode. Therefore, local modifications to the code can take effect without reinstallation.

If you have already downloaded the source code, directly enter the project root directory and run:

```shell
cd ELMERF
pip install -v -e .
```

## Optional: Install extra dependencies for development or testing

For development or testing, you may install additional dependencies:

```shell
pip install -r requirements.txt
```

For normal use, this optional step is not required.

## Step 5: Check the installation

```shell
python -c "import torch, mmcv, mmengine, mmseg; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('mmcv:', mmcv.__version__); print('mmengine:', mmengine.__version__); print('mmseg:', mmseg.__version__); print('mmseg path:', mmseg.__file__)"
```

The expected core versions are:

```shell
torch: 1.13.0+cu117
cuda: 11.7
cuda available: True
mmcv: 2.0.0
mmengine: 0.10.7
mmseg: 1.2.2
```


