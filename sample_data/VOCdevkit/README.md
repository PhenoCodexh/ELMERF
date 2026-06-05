# ELMERF VOC Test Data

This directory provides a small VOC-format test set for reproducing the
ELMERF inference and evaluation workflow.

Directory structure:

```text
VOCdevkit/
└── VOC2012/
    ├── JPEGImages/
    ├── SegmentationClass/
    ├── ImageSets/
    │   └── Segmentation/
    │       └── test.txt
    └── Annotations/
```

The test split contains 51 images and 51 semantic segmentation masks.
The `Annotations` directory is kept only for VOC-style structure. It is empty
because ELMERF evaluation uses the semantic segmentation masks in
`SegmentationClass`.

Class IDs:

- 0: background
- 1: greenshoottissues
- 2: yellowshoottissues
- 3: roots
