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
##2. Code Organization
- `dataPath` : Data folder for tranining data
- `dataPath\drivelog.csv` : drivelog.csv containing center,right and left camera image, speed, throttle, steering etc.
- `dataPath\IMG` : All the center, right and left images correspoding to filenames in drivelog.csv
- `modePath` : Folder containing saved models
- `model` : Folder with different Keras CNN models for Self Driving
- `model\VGG16.py` : VGG16 model based self driving model
- `model\nvidia.py` : Nvidia Paper based self driving model
- `model\model1.py` : Vivek blog based model
- `train.py` : Python program to train the choosen model
- `drive.py` : Python program which uses the tranined model and drive the car in the simulator autonomusly

##3. Model Building
###3.1 Approach Outline
I researched couple of models for autonumus steering prediction based on the visual image of the front facing camera
- Nvidia Paper based model
- Model purposed by Vivek in his blog
- Modified version of LeNet model
- VGG16 based model

###3.2 Data Generation & Preprocessing
Most of the the training data is straight (0 degree steering angle) from the track hence the model was overfitting for for 0 degree and driving straight even in turns. By using concepts and ideas from Vivek blog, following agumentation techniques are used
- Left and Right Camera images are used in random. The steering angles are adjusted based on the left and right camera. This give nice recovery simulation when car drift towards either of the sides
- Increase or Decrease Brightness to simulate Shadow, day and night
- Flip the center image and the negate the steering to simulate left and right turns
- Cut the image from Horixzon to top providing only bottom half for faster converting 
- Reduce image size to 64 X 64 speed up traning
- Threshold based randomization of choosing image of straight drive, this will reduce the overfitting of straight drive

The program uses a Keras fit_generator which actually can run the python generator(using Yield) in a separate thread if increase performance and memory efficient where entire processed/augmented data don't fit in the memory. The generator reads only images need for that batch and applies image pre-processing and argumentation creating a data only for that batch. It then passes this to fit_generator for training. Most of the augmentation is randomized, and the generator itself programmed to provide data continuously. Hence generator feeds data for training infinitely but very different set each time taking care of the overfitting.


###3.3 Model Architecture
###3.5 Traning
###3.6 Simulation
##4. Conclution
