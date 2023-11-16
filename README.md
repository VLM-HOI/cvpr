# VLM-HOI

## Introduction
This repository contains the code for the paper VLM-HOI: Contrastive Knowledge Distillation from the Large Vision Language Model for Human-Object Interaction Detection. The code is based on [MUREN]() and [CDN]().

<img src="figures/intro.jpg" width="500" height="auto">
<img src="figures/overview.jpg" width="1200" height="auto">

## Installation
1. Clone this repository.
2. Create conda environment using the following command:
```
conda create -n vlm_hoi python=3.9
conda activate vlm_hoi
```
3. Install the dependencies using the following command:
```
pip install -r requirements.txt
```
## Data preparation
3. Download the [data](htt)
4. Download the [pretrained model](htt)
5. Extract the data and pretrained model to:
```
VLM-HOI
├── data
│   ├── hico_20160224_det
│   └── v-coco
└── pretrained
```

## Training
### HICO-DET
To train the model, run the following command:
```
python train.py --config configs/vlm_hoi.yaml
```

### V-COCO
To train the model, run the following command:
```
python train.py --config configs/vlm_hoi_vcoco.yaml
```

## Evaluation
### HICO-DET
To evaluate the model, run the following command:
```

To evaluate the model, run the following command:
```
python test.py --config configs/vlm_hoi.yaml
```

