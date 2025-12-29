## Tutorial 2: Prepare datasets

Our dataset is organized in the form of Pascal VOC.
The specific file structure is as follows:

mmsegmentation
├── data
│   ├── RSSD
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── ImageSets
│   │   │   ├── Segmentation
|   |   |   |  ├──train.txt
|   |   |   |  ├──val.txt
|   |   |   |  ├──tset.txt

---

## Dataset Format

- **JPEGImages/**  
  This directory contains RGB images of rice seedlings collected under salt stress conditions.

- **SegmentationClass/**  
  This directory contains pixel-level semantic segmentation annotations corresponding to the RGB images.
  Each annotation is stored as a single-channel mask image, where different integer values indicate different semantic classes.

- **ImageSets/Segmentation/**  
  This directory contains text files defining dataset splits:
  - `train.txt`: list of training samples  
  - `val.txt`: list of validation samples  
  - `tset.txt`: list of test samples  

Each line in the split files corresponds to an image ID without file extension.

---

## Semantic Classes

The RSSD dataset supports organ-level semantic segmentation with the following class definitions:

| Class ID | Class Name  |
|---------:|-------------|
| 0        | Background  |
| 1        | Green leaf  |
| 2        | Yellow leaf |
| 3        | Root        |

---

## Notes

- All images in the `JPEGImages` directory must have corresponding annotation masks in the `SegmentationClass` directory.
- Image filenames and annotation filenames must be consistent.
- The dataset can be directly used with MMSegmentation by specifying the dataset root as `data/RSSD` in the configuration file.

---

## RSSD Dataset

For data acquisition, please contact  
**liuhy@hainanu.edu.cn**
