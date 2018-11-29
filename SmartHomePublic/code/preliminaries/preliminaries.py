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
import collections
import datetime
import math
import errno

import numpy as np
import numpy.linalg as alg
import pandas as pd

import matplotlib.pyplot as plt
import pylab as pylab

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from collections import defaultdict
import ast

from dateutil import parser
import tensorflow as tf
from tensorflow.python.framework import tensor_util

np.random.seed(19382)

LABEL_ENCODING = {
    "Clearing Table": 0,
    "Cooking": 1,
    "Drinking Tea": 2,
    "Eating": 3,
    "Making Tea": 4,
    "Prepping Food": 5,
    "Setting Table": 6,
    "Watching TV": 7
}

LABEL_ENCODING2NAME = {
    0: "Clearing Table",
    1: "Cooking",
    2: "Drinking Tea",
    3: "Eating",
    4: "Making Tea",
    5: "Prepping Food",
    6: "Setting Table",
    7: "Watching TV"
}
