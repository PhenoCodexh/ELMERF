# Installation and Quick Test

This guide provides three tested installation options for ELMERF:

1. Linux with GPU
2. Windows without GPU, CPU only
3. Windows with GPU

The package versions are fixed intentionally. MMCV is sensitive to the PyTorch and CUDA versions, so using the tested combinations below avoids most installation errors.

## General Note About C++ Build Tools

ELMERF depends on MMSegmentation, MMEngine, and MMCV. MMCV contains C++/CUDA extensions.

In normal installation, we recommend installing pre-built MMCV wheels. In this case, users usually do not need to compile MMCV locally.

C++ build tools are only required when MMCV must be built from source, for example when no pre-built wheel matches the local Python, PyTorch, and CUDA versions.

On Windows, install **Visual Studio Build Tools 2019 or 2022** and select:

```text
Desktop development with C++
```

The following components are recommended:

- MSVC C++ build tools
- Windows 10 SDK or Windows 11 SDK
- C++ CMake tools for Windows
- Ninja build system

After installation, open `x64 Native Tools Command Prompt for VS` and check:

```shell
cl
where cl
```

If a pre-built MMCV wheel is used, this compiler check is not normally needed.

Do not install `mmcv-lite` for ELMERF. Use full `mmcv`, because ELMERF calls `mmcv.ops`.

---

## Testing Data and Pretrained Weights

The GitHub repository provides a small VOC-format test set for installation and usage verification:

```text
sample_data/VOCdevkit/VOC2012
```

The test set contains 51 RGB images and 51 semantic segmentation masks:

```text
VOC2012/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/
│   └── Segmentation/
│       └── test.txt
└── Annotations/
```

The `Annotations` directory is kept only for VOC-style structure. It is empty because ELMERF uses the semantic segmentation masks in `SegmentationClass`.

Class IDs:

- 0: background
- 1: greenshoottissues
- 2: yellowshoottissues
- 3: roots

Place the pretrained model weight at:

```text
weight/elmerf_model.pth
```

If the weight is stored in another location, pass its path with `--checkpoint`.

The provided test script is:

```text
tools/test_elmerf_voc.py
```

It loads the pretrained model, runs inference on the VOC test split, saves predicted masks and overlay images, and reports mIoU, mAcc, and aAcc.

---

## 1. Linux GPU Installation

This is the original Linux installation used for ELMERF development and GPU testing.

Tested environment:

- Python 3.8
- PyTorch 1.13.0 + CUDA 11.7
- TorchVision 0.14.0 + CUDA 11.7
- Torchaudio 0.13.0
- MMCV 2.0.0
- MMEngine 0.10.7
- MMSegmentation 1.2.2

### Step 1: Create a Conda environment

```shell
conda create -n ELMERF python=3.8 -y
conda activate ELMERF
```

### Step 2: Install PyTorch

```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Step 3: Install OpenMMLab dependencies

```shell
pip install mmengine==0.10.7
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmsegmentation==1.2.2
```

### Step 4: Install ELMERF

```shell
git clone https://github.com/PhenoCodexh/ELMERF.git
cd ELMERF
pip install -v -e . --no-deps
```

The `-e` option installs the local source code in editable mode. Local code changes take effect without reinstalling the package.

### Step 5: Check the installation

```shell
python -c "import torch, mmcv, mmengine, mmseg; import mmcv.ops; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('mmcv:', mmcv.__version__); print('mmengine:', mmengine.__version__); print('mmseg:', mmseg.__version__)"
```

Expected key output:

```text
torch: 1.13.0+cu117
cuda: 11.7
cuda available: True
mmcv: 2.0.0
mmengine: 0.10.7
mmseg: 1.2.2
```

### Step 6: Run the VOC test set

After placing the pretrained weight at `weight/elmerf_model.pth`, run:

```shell
python tools/test_elmerf_voc.py --device cuda:0 --out-dir outputs/voc2012-test-gpu
```

The script evaluates the 51-image VOC test set and writes results to `outputs/voc2012-test-gpu`.

---

## 2. Windows CPU-Only Installation

This installation is for a Windows computer without an NVIDIA GPU, or for users who want to run ELMERF only on CPU.

We tested this environment from scratch on a Windows machine without using GPU acceleration.

Tested environment:

- Windows
- Python 3.8.19
- PyTorch 2.1.0+cpu
- TorchVision 0.16.0+cpu
- MMCV 2.1.0, CPU full wheel
- MMEngine 0.10.7
- MMSegmentation 1.2.2
- timm 1.0.27
- ftfy 6.2.3

### Step 1: Create a Conda environment

```shell
conda create -n ELMERF-cpu python=3.8 -y
conda activate ELMERF-cpu
python -m pip install --upgrade pip
```

### Step 2: Install CPU PyTorch

```shell
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install OpenMMLab dependencies and the Windows MMCV wheel

The MMCV command below installs a pre-built Windows wheel file directly from OpenMMLab. This is intentional. It avoids compiling MMCV locally on Windows.

```shell
pip install mmengine==0.10.7
pip install timm==1.0.27 ftfy==6.2.3 prettytable opencv-python pillow tqdm
pip uninstall -y mmcv mmcv-lite
pip install --force-reinstall --no-deps https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp38-cp38-win_amd64.whl
pip install mmsegmentation==1.2.2
```

This MMCV wheel is for Windows, Python 3.8, PyTorch 2.1.0, and CPU. In the wheel name, `cp38` means Python 3.8 and `win_amd64` means 64-bit Windows. If a different Python or PyTorch version is used, select the matching wheel from the MMCV installation page.

During this step, pip should download and install the `.whl` file. If pip prints `Building wheel for mmcv`, it means that pip is trying to compile MMCV from source. In that case, cancel the installation and check whether the Python, PyTorch, and MMCV wheel versions match.

### Step 4: Install ELMERF

```shell
git clone https://github.com/PhenoCodexh/ELMERF.git
cd ELMERF
pip install -v -e . --no-deps
```

If the source code has already been downloaded:

```shell
cd ELMERF
pip install -v -e . --no-deps
```

### Step 5: Check the installation

```shell
python -c "import torch, mmcv, mmengine, mmseg; import mmcv.ops; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('mmcv:', mmcv.__version__); print('mmengine:', mmengine.__version__); print('mmseg:', mmseg.__version__)"
```

Expected key output:

```text
torch: 2.1.0+cpu
cuda available: False
mmcv: 2.1.0
mmengine: 0.10.7
mmseg: 1.2.2
```

### Step 6: Run the VOC test set on CPU

After placing the pretrained weight at `weight/elmerf_model.pth`, run a quick CPU check:

```shell
python tools/test_elmerf_voc.py --device cpu --max-images 3 --out-dir outputs/voc2012-test-cpu-smoke
```

Expected key output from our Windows CPU smoke test:

```text
device=cpu
image_count=3
mIoU=0.6581
mAcc=0.9170
aAcc=0.9837
```

To test all 51 images on CPU, run:

```shell
python tools/test_elmerf_voc.py --device cpu --out-dir outputs/voc2012-test-cpu
```

CPU inference is much slower than GPU inference. On our Windows CPU test, one 1500 x 1000 image took about 13-15 seconds.

---

## 3. Windows GPU Installation

This installation is for a Windows computer with an NVIDIA GPU.

We tested this environment from scratch on a newer Windows laptop with an NVIDIA GeForce RTX 3060 Laptop GPU.

Tested environment:

- Windows
- NVIDIA driver 566.36
- NVIDIA GPU: GeForce RTX 3060 Laptop GPU
- Python 3.8.19
- PyTorch 2.1.0+cu121
- TorchVision 0.16.0+cu121
- MMCV 2.1.0, CUDA 12.1 full wheel
- MMEngine 0.10.7
- MMSegmentation 1.2.2
- timm 1.0.27
- ftfy 6.2.3

The NVIDIA driver may report a newer CUDA driver version, for example CUDA 12.7. This is acceptable. The important point is that the installed PyTorch and MMCV wheels must match each other. In this tested setup, both use CUDA 12.1 wheels.

### Step 1: Create a Conda environment

```shell
conda create -n ELMERF-gpu python=3.8 -y
conda activate ELMERF-gpu
python -m pip install --upgrade pip
```

### Step 2: Install CUDA PyTorch

```shell
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install OpenMMLab dependencies and the Windows MMCV wheel

The MMCV command below installs a pre-built Windows wheel file directly from OpenMMLab. This is intentional. It avoids compiling MMCV locally on Windows.

```shell
pip install mmengine==0.10.7
pip install timm==1.0.27 ftfy==6.2.3 prettytable opencv-python pillow tqdm
pip uninstall -y mmcv mmcv-lite
pip install --force-reinstall --no-deps https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp38-cp38-win_amd64.whl
pip install mmsegmentation==1.2.2
```

This MMCV wheel is for Windows, Python 3.8, PyTorch 2.1.0, and CUDA 12.1. In the wheel name, `cp38` means Python 3.8 and `win_amd64` means 64-bit Windows. If a different PyTorch CUDA version is used, the MMCV wheel must also be changed accordingly.

During this step, pip should download and install the `.whl` file. If pip prints `Building wheel for mmcv`, it means that pip is trying to compile MMCV from source. In that case, cancel the installation and check whether the Python, PyTorch CUDA, and MMCV wheel versions match.

### Step 4: Install ELMERF

```shell
git clone https://github.com/PhenoCodexh/ELMERF.git
cd ELMERF
pip install -v -e . --no-deps
```

If the source code has already been downloaded:

```shell
cd ELMERF
pip install -v -e . --no-deps
```

### Step 5: Check the installation

```shell
python -c "import torch, mmcv, mmengine, mmseg; import mmcv.ops; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('mmcv:', mmcv.__version__); print('mmengine:', mmengine.__version__); print('mmseg:', mmseg.__version__)"
```

Expected key output:

```text
torch: 2.1.0+cu121
cuda: 12.1
cuda available: True
mmcv: 2.1.0
mmengine: 0.10.7
mmseg: 1.2.2
```

### Step 6: Run the VOC test set on GPU

After placing the pretrained weight at `weight/elmerf_model.pth`, run:

```shell
python tools/test_elmerf_voc.py --device cuda:0 --out-dir outputs/voc2012-test-gpu
```

Expected key output from our Windows GPU test:

```text
device=cuda:0
cuda_device=NVIDIA GeForce RTX 3060 Laptop GPU
image_count=51
mIoU=0.6311
mAcc=0.9171
aAcc=0.9832
```

In our Windows GPU test, all 51 images of size 1500 x 1000 were processed successfully. The full run took about 31 seconds on an RTX 3060 Laptop GPU, with peak GPU memory usage of about 2.4 GB.

---

## Common Problems

### `No module named 'mmcv._ext'`

This usually means that `mmcv-lite` was installed, or that the MMCV wheel does not match the PyTorch version.

Fix:

```shell
pip uninstall -y mmcv mmcv-lite
```

Then reinstall the matching full `mmcv` wheel for your platform.

### `torch.cuda.is_available()` is `False` on a GPU computer

Check the following:

- The NVIDIA driver is installed.
- The installed PyTorch package is a CUDA build, for example `torch==2.1.0+cu121`.
- The code uses `device='cuda:0'`.

### Many `missing keys` or `unexpected keys` when loading a checkpoint

Do not ignore this message. It means that part of the model weights may not have been loaded. Check that the checkpoint, config file, and source code come from the same ELMERF version.

### pip tries to build MMCV from source

On Linux, MMCV can often be installed directly with pip or mim because a compatible pre-built wheel is available.

However, MMCV contains C++/CUDA extensions. If pip cannot find a wheel matching the local Python, PyTorch, CUDA, and operating system versions, it may print `Building wheel for mmcv` and try to compile MMCV locally. In that case, C++ build tools are required.

For most users, especially on Windows, we recommend installing the specified pre-built MMCV wheel instead of building MMCV from source.
