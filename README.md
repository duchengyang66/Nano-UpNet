# Nano-UpNet
# Nano-UpNet

Nano-UpNet is a deep-learning framework for predicting nanoparticle uptake from label-free bright-field microscopy images.

The inference pipeline consists of two steps:

1. **Cell-type recognition using YOLOv8**
2. **Nanoparticle uptake prediction using pix2pixHD**

## Installation

Clone this repository:

```bash
git clone https://github.com/duchengyang66/Nano-UpNet.git
cd Nano-UpNet
```

Install the required dependencies according to the corresponding model implementations.

## Usage

### Step 1: Cell-type recognition

Run the following script:

```bash
python cell_type_recognition/ultralytics/yolo_prepare.py
```

This step uses the YOLOv8-based model to process bright-field microscopy images and generate **cell-type labels**.

The generated labels are used by the subsequent uptake prediction step.

### Step 2: Nanoparticle uptake prediction

After generating the cell-type labels, run:

```bash
python cell_uptake_prediction/pix2pixHD/inference.py
```

This step uses the pix2pixHD-based model to predict **nanoparticle uptake images** from the bright-field microscopy images.

## Workflow

```text
Bright-field image
       │
       ▼
yolo_prepare.py
       │
       ▼
Cell-type labels
       │
       ▼
inference.py
       │
       ▼
Predicted nanoparticle uptake
```

## References

This project uses the following open-source implementations:

* [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
* [NVIDIA pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

Please refer to the original repositories for the corresponding model implementations and licenses.