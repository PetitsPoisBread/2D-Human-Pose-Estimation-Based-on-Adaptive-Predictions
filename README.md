# 2D Human Pose Estimation Based on Adaptive Predictions

This is the implementation of the paper "2D Human Pose Estimation Based on Adaptive Predictions"

## Dependencies
- Pytoch 1.8.1
- Numpy
- Scipy
- Pycocotools
- openCV
## Dataset
**MS-COCO 2017**

please download the data from the following link

Link: [trajectories](https://cocodataset.org/#home)

## Directory Structure
.

├── tools        -- Including tools such as training and testing

├── experiments  -- config files

├── lib          -- Including third-party libraries

├── data         -- Dataset directory

├── README.md          

└── train.sh     -- Main file for training / inference
## Training
To train a model from scratch you should look up the model's configuration options.

Here is one example:

`sh train.sh`
