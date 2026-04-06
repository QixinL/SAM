# SAM Experiments Repository

This repository is for the project `Evaluating the Effect of Sharpness-Aware Minimization on Model Generalization`.

It contains two experiment tracks:

- CIFAR-10
- GTSRB

The experiments are based on Sharpness-Aware Minimization (SAM) from Foret et al. [1].

This repository contains multiple experiment branches. Start on `main`, then switch to the branch for the workflow you want.

## Branches

### CIFAR-10 experiment

Switch to the CIFAR-10 branch:

```bash
git switch CIFAR-10
```

Then run:

```bash
python main.py
python main_sam.py
```

### GTSRB experiment

Switch to the GTSRB branch:

```bash
git switch GTSRB
```

Then follow the branch-specific README for the German traffic sign training workflow.

### German dataset visualization

Switch to the data exploration branch:

```bash
git switch Data_Exploration
```

Then run:

```bash
python plot_data_second_class_distribution.py
```

## Citation

[1] P. Foret, A. Kleiner, H. Mobahi, and B. Neyshabur, Sharpness-Aware Minimization for Efficiently Improving Generalization, in International Conference on Learning Representations (ICLR), 2021.

