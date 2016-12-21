import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.callbacks import Callback
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D