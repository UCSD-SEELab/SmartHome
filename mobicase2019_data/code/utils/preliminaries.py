from __future__ import print_function
import os
import re
import sys
import json
import time
import datetime
import shutil
import tempfile
import traceback
import matplotlib
import datetime
import math
import errno

import numpy as np
import numpy.linalg as alg
import pandas as pd

import matplotlib.pyplot as plt
import pylab as pylab

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import xgboost


from scipy.stats import kurtosis
from scipy.stats import skew

from collections import defaultdict
import ast

import cv2
from dateutil import parser
from termcolor import colored
import tensorflow as tf

np.random.seed(19382)
