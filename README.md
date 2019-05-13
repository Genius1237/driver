# Self Driving Cars in Video Games using Deep Learning

### Contents
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Collecting Data for IL](#Collecting-Data-for-IL)
- [Training IL](#Training-IL)
- [License](#License)

### Introduction
The aim of of this project is to experiment with self driving cars in video games using a number of different techniques. 
Grand Theft Auto V has been chosen as a platform to experiment on. The game, released in 2015 by Rockstar games is one of the most popular open-world games in existance.

The current aim is to train the agent using Imitation learning. Future goals are to use Reinforcement Learning (something along the lines of policy gradient) and Generative Adversarial Imitation Learning (GAIL).

This project is heavily inspired by the self driving car project by [sentdex](https://github.com/sentdex). His [GitHub repo](https://github.com/sentdex/pygta5) has been referred to quite frequently in this project.

### Installation
This project is using `python3`. Please make sure that your environment is running Python 3.6 or newer.

This model is built in top of `PyTorch`, follow the installation instructions from PyTorch [here](https://pytorch.org/get-started/locally/).

To install dependencies, do
```bash
pip install -r requirements.txt
```

### Collecting Data for IL
Since most video games require **DirectX** and run on **Windows** only, this process is limited to Windows.

`grabscreen.py` uses an efficient method to capture the screen on a window desktop. This is using `win32gui` that is part of the `pywin32` package. This method is from [Stackoverflow](https://stackoverflow.com/questions/3586046/fastest-way-to-take-a-screenshot-with-python-on-windows).

In it's current state, it captures a crop of the top left corner of the screen, which is then resized to a resolution of 640x360 to be used by the model. Ideally, you want the video game to run in windowed mode in a resolution of 1066x600 and positioned in the top left corner. The windows title bar (which is 40 pixels in my case) is accounted for while cropping the image.

Key presses are captured in `getkeys.py`. This is also using a library that is part of the `pywin32` package.

To capture data and store it to a `HDF5` file, run
```bash
python screen_capture.py [--file OUTPUT_FILE]
                         [--window-resolution WINDOW_HEIGHT WINDOW_WIDTH]
                         [--titlebar-adjustment TITLEBAR_ADJUSTMENT]
                         [--save-test-image]
```
Once it's running, switch over to the video game and press X to start the capture. Press Y to stop the capture. The data is written to a `.hdf5` file. 

The `hdf5` file will have 4 datasets within it.
* `train_X` and `train_Y` that can be used for training
* `test_X` and `test_Y` that can be used as test or validation data

`train`:`test` is in a 4:1 ratio.

Some training data has been collected and is avaiable [here](https://drive.google.com/drive/folders/1iEES6uvdd8H5oWW5aMQcqBvfai9NIRpV?usp=sharing). The folder will include all datasets collected so far.

### Training IL
Unlike RL based methods, which will probably have to run on **Windows** only (since episodes have to be generated on the fly), this part can be done on any operating system.

The model is a CNN based architecture. Currently, `AlexNet` is being used (because my 1050Ti cannot fit anything bigger). It is trained using the simple behaviour cloning technique, which has a lot of pitfalls that need to be addressed. 

To train the model,
```bash
python train_il.py [--file INPUT_FILE] [--model MODEL_NAME]
                   [--batch-size BATCH_SIZE] [--n-epochs N_EPOCHS]
                   [--device DEVICE]
```
We are yet to train the model to view how the performance is. We expect a data-imbalance problem to be there and a solution for that has to be looked into.

### License
This project uses the [MIT License](https://opensource.org/licenses/MIT)
