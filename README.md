# GTSRB Experiments

This branch trains and compares a standard ResNet-18 baseline and a Sharpness-Aware Minimization (SAM) ResNet-18 on the German Traffic Sign Recognition Benchmark (GTSRB), using a leakage-safe `DATA_second` workflow.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Create a folder named `DATA_second/` at the repository root and place the GTSRB data there.

This code expects the dataset to contain:

- `DATA_second/Train.csv`
- `DATA_second/Test.csv`
- the image files referenced by those CSVs

The training code builds a clean split from the original training set only, keeps the original test set held out, and removes exact train images that also appear in test.

## Recommended Run Order

Run the models in this order:

1. Train the baseline model first.
2. Train the SAM model second.
3. Evaluate checkpoints only after training is complete.

This keeps the baseline as the reference run and makes the SAM comparison easier to interpret.

## Train The Baseline

```bash
python train_data_second_baseline.py
```

By default this trains on the clean `DATA_second` split with:

- `lr=0.019920`
- `momentum=0.8132`
- `weight_decay=0.000038`
- `batch_size=64`
- `epochs=20`
- `train_target_per_class=270`

The baseline trainer uses training-time augmentation and records per-epoch:

- train accuracy
- validation accuracy
- train loss
- validation loss

It also saves the best validation checkpoint.

## Train The SAM Model

```bash
python train_data_second_sam.py
```

The SAM trainer uses the same baseline hyperparameters for:

- learning rate
- momentum
- weight decay
- batch size
- epochs
- train target per class

It also tracks per-epoch train/validation accuracy and loss and saves the best validation checkpoint.

## Evaluate A Trained Checkpoint

Baseline:

```bash
python evaluate_data_second_baseline.py --checkpoint <path-to-checkpoint> --split-dir <path-to-split-dir> --split val
```

SAM:

```bash
python evaluate_data_second_sam.py --checkpoint <path-to-checkpoint> --split-dir <path-to-split-dir> --split val
```

Use `--split val` if you want validation evaluation. Use `--split test` only when you are ready for held-out test evaluation.

## Notes

- The clean split is created from `Train.csv`; the original `Test.csv` remains the held-out test set.
- Training augmentation is applied on the fly during dataloader construction; it does not write augmented images back to disk.
- Validation and test preprocessing remain deterministic.

## Citation

The data is free to use. However, please cite the following publication if you use it:

J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453-1460. 2011.

```bibtex
@inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    year = {2011},
    pages = {1453--1460}
}
```
