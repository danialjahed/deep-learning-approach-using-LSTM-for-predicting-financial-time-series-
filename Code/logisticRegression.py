import keras as k
import tensorflow as tf 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0, solver='sag')