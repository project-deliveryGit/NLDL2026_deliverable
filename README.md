# Domain Adaptation for Phantom Ultrasound Vessel Segmentation

Anonymous submission — NLDL 2026 Winter School Project

This repository contains code for adapting a U-Net segmentation model trained on real patient 
laparoscopic ultrasound to phantom ultrasound recordings. Three domain adaptation strategies 
are evaluated: fine-tuning, LoRA, and cross-attention.

All models were trained using grid search over their respective hyperparameters. The best 
configuration from each grid search was selected based on validation Dice score and evaluated 
on a held-out test set to produce the final results. No data or pretrained model weights are 
shared in this repository.

This project was conducted as part of the NLDL 2026 Winter School. The cross-attention approach 
was implemented following methods presented in one of the school's tutorials. Fine-tuning and 
LoRA adaptations were inspired adapted based on prior machine learning coursework at the authors' institution.

---

## Repository Structure
```
.
├── config/
│   ├── finetune.yaml
│   ├── lora.yaml
│   └── cross_attention.yaml
├── src/
│   ├── utils/
│   │   ├── dataset.py          # Dataset, sampler, train/val split
│   │   ├── metrics.py          # Dice, IoU, recall, precision
│   │   ├── models.py           # All model definitions and loaders
│   │   └── training.py         # Shared training loop and helpers
│   ├── training/
│   │   ├── grid_search_finetune.py
│   │   ├── grid_search_lora.py
│   │   └── grid_search_cross_attention.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── visualization/
│       └── domain_shift.py
├── requirements.txt
└── README.md
```

---

## Requirements
```bash
pip install -r requirements.txt
```


## Usage

### Fine-tuning grid search
```bash
python src/training/grid_search_finetune.py \
```

### LoRA grid search
```bash
python src/training/grid_search_lora.py \
```

### Cross-attention grid search
```bash
python src/training/grid_search_cross_attention.py \
```

### Evaluation
```bash
python src/evaluation/evaluate.py \
```

### Domain shift visualization
```bash
python src/visualization/domain_shift.py \
```


## Data

The expected directory structure for the phantom dataset is:

```
labeled/
├── US-Acq_1_.../
│   ├── preprocessed/
│   │   ├── frame_000001.png
│   │   └── ...
│   └── masks_edited/
│       ├── mask_000001.png
│       └── ...
└── US-Acq_8_.../     ← held-out test case
    ├── preprocessed/
    └── masks_edited/
...
```
