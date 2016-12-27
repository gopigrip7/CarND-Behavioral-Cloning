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
python drive.py models/model.json
```
###1.3 Training the model
The model can be tranined from scratch or from previsouly trained model
- Training from Scratch
```cmd
python train.py --dpath data/drivelog.csv --epoch 8 --mpath model/
```
- Training already tranined model
```cmd
python train.py --dpath data/drivelog.csv --epoch 8 --mpath model/modelv1.json --restore
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
The problem is to get continuous ouput which is different from the classifer, this problem should be considered as Regresstion. 

###3.2 Data Generation & Preprocessing
Most of the the training data is straight (0 degree steering angle) from the track hence the model was overfitting for for 0 degree and driving straight even in turns. By using concepts and ideas from Vivek blog, following agumentation techniques are used
- Left and Right Camera images used in random. The steering angles are adjusted based on the left and right camera. This give nice recovery simulation when car drift towards either of the sides
- Increase or Decrease Brightness to simulate Shadow, day and night
- Flip the center image and negate the steering to simulate left and right turns
- Cut the image from Horixzon to top providing only bottom half for faster converting 
- Reduce image size to 64 X 64 speed up traning
- Threshold based randomization of choosing image of straight drive, this will minimize the overfitting of straight drive

The program uses a Keras fit_generator which actually can run the python generator(using Yield) in a separate thread if increase performance and memory efficient where entire processed/augmented data don't fit in the memory. The generator reads only images need for that batch and applies image pre-processing and argumentation creating a data only for that batch. It then passes this to fit_generator for training. Most of the augmentation is randomized, and the generator itself programmed to provide data continuously. Hence generator feeds data for training infinitely but very different set each time taking care of the overfitting.

###3.3 Model Architecture
As briefied in apporach outline, after trying out various models implemented Vivek's model which gave good performace. Below is the summary of the Model Architecture from Keras.Summary().

Model uses Lambda layer to do in model nomalization( moving the 0 255 color space to 0 to 1. Instead of gray scale or any color channel, the model first convolution layer 3 channel depth will make best color feature space based on the training data. The first color channel is followed by  3 more layers of convolution with channel 32, 64, 64, 64 and the kernel 3x3. each of the convolution followed by 2x2 maxpooling. Followed by Flatten and Dense layer of 512, 64, 16 and 1. As stated above predicting steering wheel from image is regression problem to continuously predict a steering from the given image, hence the last dense layer is 1 without any activation or softmax layer.

Leaky ReLUs allow a small, non-zero gradient when the unit is not active thus providing smooth steering angle between images.

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to unscroll output; double click to hide" style="display: block;"></div><div class="output output_scroll" style="display: flex;"><div class="output_area"><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 3)     12          lambda_1[0][0]                   
____________________________________________________________________________________________________
leakyrelu_1 (LeakyReLU)          (None, 64, 64, 3)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 64, 32)    896         leakyrelu_1[0][0]                
____________________________________________________________________________________________________
leakyrelu_2 (LeakyReLU)          (None, 64, 64, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 32, 32)    0           leakyrelu_2[0][0]                
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 32, 32)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 32, 32, 64)    18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
leakyrelu_3 (LeakyReLU)          (None, 32, 32, 64)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 16, 64)    0           leakyrelu_3[0][0]                
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 16, 16, 64)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 16, 16, 64)    36928       dropout_2[0][0]                  
____________________________________________________________________________________________________
leakyrelu_4 (LeakyReLU)          (None, 16, 16, 64)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 8, 8, 64)      0           leakyrelu_4[0][0]                
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 8, 8, 64)      0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4096)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           2097664     flatten_1[0][0]                  
____________________________________________________________________________________________________
leakyrelu_5 (LeakyReLU)          (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 512)           0           leakyrelu_5[0][0]                
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 64)            32832       dropout_4[0][0]                  
____________________________________________________________________________________________________
leakyrelu_6 (LeakyReLU)          (None, 64)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 64)            0           leakyrelu_6[0][0]                
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            1040        dropout_5[0][0]                  
____________________________________________________________________________________________________
leakyrelu_7 (LeakyReLU)          (None, 16)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 16)            0           leakyrelu_7[0][0]                
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dropout_6[0][0]                  
====================================================================================================
Total params: 2187885
____________________________________________________________________________________________________
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;"></div></div>
###3.5 Training
The Training uses Adam optimiser with default parameters which work best. Considering the regression output explained above, MSE loss (Mean Square root) function employed in the training process. Any epoch greater than 8 the model seems to overfit and mostly drives straight, the assumptions is due to the nature of prediction problem and images. 

Dropout of 0.5 is used to regularise and avoid overfitting, this drops the weight in random by half there by reduce overfitting and also allow other neuron to learn the feature.

The model was run with batch size of 250 and samples_per_epoch 20000.

The test, validation and train split is 10:10:80 and model uses the same generator to perform the validation. The validation loss decrease around 8 and increase after that due to nature of the problem. Thought testing dataset was used but it doesn't seem to provide valid test, and hence the testing was actually done in the autonomous mode

The entire model training takes around 6 to 8 mins. Both the model architecture and model weights are stored in model folder provided in the command line or default in the current folder.

###3.6 Simulation
The drive.py which exports the model and drives in autonomus mode uses the same preprocessing and image pipeline before feeding the image into prediction function. The result of predicted steering angle is divide by 1.25; this provides best smooth turns without crossing the lines. The throttle set to 0.2, any increase in throttle the car drives out of the safety zone.
##4. Conclution
Thought the project goal is met, it is far from over. Nvidia model or the method I implmented from vivek's blog all consider image and steering prediction as discrent problem rather than a continuos problem in action domain. Example like catching a blow thrown. The ultimate autonimus can only be achived when these problems are consider in action domain, second in failure of camera or any interption , model would have already predict to extent the steering values to drive to safety.
Hence this problem needs to be approached using RNN  concept as continuous prediction. The 1st among the Challenge 1 used RNN which would be right path, but I am yet explore the same.


