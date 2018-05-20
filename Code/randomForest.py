import keras as k
import tensorflow as tf 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(max_depth=20,n_estimators=1000,random_state=42)