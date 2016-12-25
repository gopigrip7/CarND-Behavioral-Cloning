# CarND-Behavioral-Cloning
## Project Overview
The Project is the third in Udacity SelfDriving CarND. The goal is to clone the human car driving behavior using DeepLearning Technique and replay it to drive (Only Steering) the car autonomously. For this Udacity has provided Simulator for recording the human driving behavior and an autonomous option to drive using the cloned behavior using deepLeanring.

##1. Quick start
###1.1 Pre-request
- Tensorflow
- Keras
- Numpy
- Pandas
- OpenCV3

###1.2 Self Drive in Autonomus mode
Start the simulator in an autonumus mode and run the following command in terminal
```cmd
python drive.py model/model.json
```
###1.3 Training the model
The model can be tranined from scratch or from previsouly trained model
- Training from Scratch
```cmd
python train.py --dpath data/drivelog.csv --epoch 8 --mpath model/
```
- Training already tranined model
```cmd
python train.py --dpath data/drivelog.csv --epoch 8 --mpath model/ --restore
```
##2 Code Organization
- `dataPath` : Data folder for drivelog.csv containing center,right and left camera image, speed, throttle, steering etc.
- `dataPath\IMG` : All the center, right and left images correspoding to filenames in drivelog.csv
- `modePath` : Folder containing saved models
- `model` : Folder with different Keras CNN models for Self Driving
- `model\VGG16.py` : VGG16 model based self driving model
- `model\nvidia.py` : Nvidia Paper based self driving model
- `model\model1.py` : Vivek blog based model

