import numpy as np 
import pandas as pd 
import scipy 
import tensorflow as tf 
import keras as k 
import pickle as pk
import os

from generateWindows import Windowing 
from ModelEval import ModelEval 

from DNN import model as DNNmodel








if __name__ == '__main__':


    pklAddress = '../DataSets/windows/generateWindows.pickle'
    windows = Windowing()
    windows.loadFromPickle(pklAddress)

    eModel = ModelEval(windowingModel=windows)
    eModel.loadFromPickle()


    eModel.kerasEvalModel(DNNmodel)
    print(eModel.metrics)

