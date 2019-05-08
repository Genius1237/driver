## Self Driving Cars in Video Games using Deep Learning

### Contents
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Input](#Input)
- [Usage](#Usage)

### Introduction
The aim of of this project is to experiment with self driving cars in video games using a number of different techniques. 
Grand Theft Auto V has been chosen as a platform to experiment on. The game, released in 2015 by Rockstar games is one of the most popular open-world games in existance.

The current aim is to train the agent using Imitation learning. Future goals are to use Reinforcement Learning (something along the lines of policy gradient) and Generative Adversarial Imiation Learning (GAIL).

### Installation
This model is built in top of PyTorch, follow the installation instructions from PyTorch [here](https://pytorch.org/get-started/locally/).

To install dependencies, do
```bash
pip install -r requirements.txt
```

### Input
Currently, only the **Windows OS** is supported due to the `win32gui` requirements for screen grabbing / input capture.
The `screen_capture.py`, `grabscreen.py`, `directkeys.py` and `getkeys.py` scripts handle screen capturing, image grabbing and keypress event recording.

### Usage
```bash
python train_il.py --data <path-to-train-data> --model <name-of-model-architecture> --batch-size 
<batch-size-for-train> --n-epochs <no-of-epochs> --device <cpu/cuda>
```
which trains the model for Imitation Learning.

