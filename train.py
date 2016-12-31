import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from generator import Generator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import scipy
import cv2
import re

from model import Model

np.random.seed(45)

# parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--dPath", dest="data_path", default="data",action="store") 
parser.add_argument("--mPath", dest="model_path", default="models",action="store")
parser.add_argument("--restore", dest="restore", default = False,action="store_true")
parser.add_argument("--nb_epoch", dest="nb_epoch", default = 8, type=int)
parser.add_argument("--model",default="custModel",choices=["nvidiaModel","custModel"],action="store")
parser.add_argument("--rModel",dest="restore_model",default="models/model.json",action="store")
parser.add_argument("--oModel",dest="output_model",default="model",action="store")
args = parser.parse_args()
print(args)

# printing All Agruments for Traning
print("You have selected following Parameter for Training")
print("-"*50)
print("Model will be saved in Path              : {}\\".format(args.model_path))
print("Traning data path                        : {}\\".format(args.data_path))
print("Model Type                               : {}".format(args.model))
if args.restore:
    print("You have asked to Restore model")
    print("Model choose to restore                  : {}".format(args.restore_model)) 
print("Output model files                       : {}".format(args.output_model))
print("Number of training Epochs                : {}".format(args.nb_epoch))

input_shape =(64,64,3) # Input shape of size 64x64 with 3 color channels

if args.restore:
    print("Restoring model from {}".format(args.restore_model))
    with open(args.model, 'r') as jfile:
        drive_model = model_from_json(json.load(jfile))

    drive_model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    drive_model.load_weights(weights_file)
else:
    drive_model = Model(input_shape, args.model).getModel() # Creating model give shape and model. Model1 
    drive_model.compile("adam", "mse")


print("Train Model")
generator = Generator(input_shape,args.data_path)
for ep in range(args.nb_epoch):
    gen_train = generator.generate_data()
    gen_valid = generator.generate_data()
    hist = drive_model.fit_generator(gen_train,samples_per_epoch = 20000, nb_epoch=1, verbose=1
                           ,validation_data=gen_valid,nb_val_samples=2000)
    
with open("{}\{}.json".format(args.model_path,args.output_model), 'w') as outfile:
    json.dump(drive_model.to_json(), outfile)
drive_model.save_weights("{}\{}.h5".format(args.model_path,args.output_model))