# SAM

Reimplementation of Sharpness Aware Minimization (SAM) https://arxiv.org/pdf/2010.01412.
We compare the performance of SAM with Stochastic Gradient Descent (SGD) on CIFAR-10 and GTSRB on smaller parameter models (ResNet-18) that can be trained locally.

### Setup

git clone https://github.com/QixinL/SAM.git
cd SAM

```bash
pip install -r requirements.txt
```

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
## Current Status
main.py runs SGD on CIFAR-10
main_sam.py runs SAM on CIFAR-10

Currently, training on GTSRB is in a seperate branch (called GTSRB)

## Project Structure

```
src/
  ├── model.py     # ResNet-18
  ├── model_sam.py # Instanciates model.py for SAM
  ├── data.py      # CIFAR-10 dataset loading
  ├── train.py     # Trains SGD
  └── sam_train.py # Trains SAM 

main_sam.py        # Runs SAM training
main.py            # Runs SGD training
MIE424_Dataset     # Test for loading CIFAR-10 and GTSRB
requirements.txt
README.md
```

