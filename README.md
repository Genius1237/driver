## Self Driving Cars in Video Games using Deep Learning

### Contents
- [Installation](#Installation)
- [Usage](#Usage)


### Installation
This model is built in top of PyTorch, follow the installation instructions from PyTorch [here](https://pytorch.org/get-started/locally/)
Currently, only the **Windows OS** is supported due to the `win32gui` requirements for screen grabbing / input capture.

To install dependencies, do
```bash
pip install -r requirements.txt
```

### Usage
```bash
python train_il.py --data <path-to-train-data> --model <name-of-model-architecture> --batch-size 
<batch-size-for-train> --n-epochs <no-of-epochs> --device <cpu/cuda>
```
which trains the model for Imitation Learning
