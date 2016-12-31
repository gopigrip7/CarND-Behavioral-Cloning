import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

from keras.callbacks import Callback
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU,Activation,BatchNormalization,LeakyReLU,ELU
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import scipy
import re

np.random.seed(45)

class Model:
    def __init__(self,input_shape,model="nvidiaModel",keep_prob=0.5):
        self.input_shape = input_shape
        self.model = model
        self.keep_prob = keep_prob
        self.models={"nvidiaModel" : self.getNvidia,"custModel" :self.getcustModel}
    
    def getModel(self):
        return self.models[self.model](self.input_shape,self.keep_prob)
    
    def getNvidia(self,input_shape,keep_prob):
        ch,col,row =input_shape
        model = Sequential()
        model.add(Lambda(
              lambda x: x/255.,
              input_shape=input_shape,
              output_shape=input_shape)
          )
        model.add(BatchNormalization(input_shape=input_shape))
        model.add(Convolution2D(3, 5, 5,subsample=(2,2), border_mode="same",init="he_normal"))
        model.add(Activation('relu'))
        model.add(Convolution2D(24, 5, 5,subsample=(2,2), border_mode="same",init="he_normal"))
        model.add(Activation('relu'))
        model.add(Convolution2D(36, 5, 5,subsample=(2,2), border_mode="same",init="he_normal"))
        model.add(Activation('relu'))
        model.add(Convolution2D(48, 3, 3,subsample=(1,1), border_mode="same",init="he_normal"))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3,subsample=(1,1), border_mode="same",init="he_normal"))
        model.add(Flatten())
        model.add(Dense(1164,init='uniform'))
        model.add(Dropout(keep_prob))
        model.add(Activation('relu'))
        model.add(Dense(100,init='uniform'))
        model.add(Dropout(keep_prob))
        model.add(Activation('relu'))
        model.add(Dense(50,init='uniform'))
        model.add(Dropout(keep_prob))
        model.add(Activation('relu'))
        model.add(Dense(10,init='uniform'))
        model.add(Activation('relu'))
        model.add(Dense(1))

        return model


    def getcustModel(self,input_shape,keep_prob):

        model = Sequential()
        model.add(Lambda(
          lambda x: x/255.,
          input_shape=input_shape,
          output_shape=input_shape)
      )
        model.add(Convolution2D(3, 1, 1, border_mode="same",init="he_normal"))
        model.add(LeakyReLU())


        model.add(Convolution2D(32, 3, 3, border_mode="same",init="he_normal")) 
        model.add(LeakyReLU())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(keep_prob))

        model.add(Convolution2D(64, 3, 3, border_mode="same",init="he_normal"))
        model.add(LeakyReLU())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(keep_prob))

        model.add(Convolution2D(64, 3, 3, border_mode="same",init="he_normal"))
        model.add(LeakyReLU())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(keep_prob))

        model.add(Flatten())

        model.add(Dense(512,init='uniform'))    
        model.add(LeakyReLU())
        model.add(Dropout(keep_prob))

        model.add(Dense(64,init='uniform'))
        model.add(LeakyReLU())
        model.add(Dropout(keep_prob))

        model.add(Dense(16,init='uniform'))
        model.add(LeakyReLU())
        model.add(Dropout(keep_prob))

        model.add(Dense(1))


        return model