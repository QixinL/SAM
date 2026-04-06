# CIFAR-10 SAM vs SGD

This repository compares standard SGD against Sharpness-Aware Minimization (SAM) on CIFAR-10 using a ResNet-18 model.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

The CIFAR-10 dataset is downloaded automatically by `torchvision` the first time you run an experiment.

## Repository Layout

- `main.py`: baseline SGD experiment entry point
- `main_sam.py`: SAM experiment entry point
- `src/data.py`: CIFAR-10 loaders and augmentation
- `src/model.py`: ResNet-18 for CIFAR-10
- `src/train.py`: baseline SGD training loop
- `src/sam_train.py`: SAM optimizer and SAM training loop

## Exact Run Order

Run these files in this order:

1. Run `main.py` to train the baseline SGD model on CIFAR-10.
2. Run `main_sam.py` to train the SAM model on CIFAR-10.

Commands:

```bash
python main.py
python main_sam.py
```

### File 1: `main.py`

```bash
python main.py
```

This file:

- loads CIFAR-10 through `src/data.py`
- builds the ResNet-18 model from `src/model.py`
- trains the baseline with `src/train.py`

Default settings:

- `seed=42`
- `batch_size=64`
- `epochs=200`
- `lr=0.1621948163070703`
- `momentum=0.9285729026103532`
- `weight_decay=0.0016196219073369704`
- `train_fraction=0.1`
- `val_split=0.1`

### File 2: `main_sam.py`

```bash
python main_sam.py
```

This file:

- loads CIFAR-10 through `src/data.py`
- builds the model through `src/model_sam.py`
- trains with SAM through `src/sam_train.py`

The SAM run uses the same base hyperparameters as the baseline, with the additional SAM parameter:

- `rho=0.05`

## What The Code Does

- The model is a CIFAR-10 ResNet-18.
- The loader uses a subset of the CIFAR-10 training set controlled by `train_fraction`.
- The training split uses `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()`.
- Validation and test splits use normalization only.
- Both training scripts report train loss, validation loss, and validation accuracy during training.
- The current tracked entry points train against the validation split; they do not save checkpoints by default.

## Command Examples

Run the exact baseline file with custom settings:

```bash
python main.py --epochs 100 --batch-size 128 --train-fraction 1.0
```

Run the exact SAM file with custom settings:

```bash
python main_sam.py --epochs 100 --batch-size 128 --train-fraction 1.0 --rho 0.05
```

You can override:

- `--seed`
- `--batch-size`
- `--epochs`
- `--lr`
- `--momentum`
- `--weight-decay`
- `--train-fraction`
- `--val-split`
- `--device`

SAM also supports:

- `--rho`

## Notes

- On Windows, the dataloader uses `num_workers=0` to avoid worker hangs.
- If CUDA is available, the scripts use it automatically unless you override `--device`.
