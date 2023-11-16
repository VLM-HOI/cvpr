# VLM-HOI

<img src="figures/intro.jpg" width="400" height="auto">
<img src="figures/overview.jpg" width="1200" height="auto">

## Introduction
This repository contains the code for the paper VLM-HOI: Contrastive Knowledge Distillation from the Large Vision Language Model for Human-Object Interaction Detection. The code is based on [MUREN]() and [CDN]().

## Installation
1. Clone this repository.
2. Install the dependencies using the following command:
```
pip install -r requirements.txt
```
3. Download the [data](htt)
4. Download the [pretrained model](htt)

## Training
To train the model, run the following command:
```
python train.py --config configs/vlm_hoi.yaml
```

## Evaluation
To evaluate the model, run the following command:
```
python test.py --config configs/vlm_hoi.yaml
```

