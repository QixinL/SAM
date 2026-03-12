# SAM vs Baseline Comparison

Minimal working version to test Sharpness Aware Minimization (SAM) vs standard SGD on CIFAR-10.

## Current Status

✓ CIFAR-10 data loading (10% subset)
✓ ResNet-18 model
- [ ] SAM optimizer
- [ ] Training loops
- [ ] Experiment runner

## Quick Test

```bash
pip install -r requirements.txt
python test.py
```
This loads CIFAR-10 (5k training samples) and runs a forward pass through ResNet-18.

Optional: venv
```bash
python -m venv venv
```

Optional: CUDA
```bash
# If you want CUDA and have 3.11 or 3.12 installed
py -3.12 -m venv venv
nvidia-smi # Obtain your CUDA Version (eg: 13.2) and ask gpt or google to give you the whl installation

# If version newer than 12.8, use 12.8 (latest version) using:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Project Structure

```
src/
  ├── model.py     # ResNet-18
  ├── data.py      # CIFAR-10 loading
  ├── train.py     # Implemented baseline. Todo: implement sam
  └── (sam.py)     # Coming next

test.py            # Validation script
requirements.txt
README.md
```

## Next Steps

1. Add SAM optimizer
2. Add training loops  
3. Create experiment runner for 5 replicas

