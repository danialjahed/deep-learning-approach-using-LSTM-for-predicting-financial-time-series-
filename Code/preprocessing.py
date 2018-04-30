import pandas as pd 
import numpy as np 
import scipy as sp
import loadData


def calcReturn(inputData):
    Return = [0]
    inputLen = inputData.shape[0]
    for i in range(1,inputLen) :
        Return.append( (inputData['close'][i]/inputData['close'][i-1]) - 1 )

    return np.array(Return)

def standardizeReturn(inputData):
    std = np.std(inputData)
    mean = np.mean(inputData)

    Sreturn = (inputData-mean)/std

    return Sreturn


def classify(inputData):
    median = np.median(inputData)
    comp = lambda x,y: x>y
    vComp = np.vectorize(comp)
    Y = vComp(inputData,median)
    Y = 1 * Y
    return Y


def preprocess(inputDataFrame):

    r = calcReturn(inputDataFrame)
    # print(r)
    # add columns to inputdataframe
    inputDataFrame.loc[:,'return'] = r
    # print('******************')
    
    sr =standardizeReturn(r)
    # print(sr)
    # add columns to inputdataframe
    inputDataFrame.loc[:,'Sreturn'] = sr
    # print('******************')
    
    y = classify(sr)
    # print(y)
    # add columns to inputdataframe
    inputDataFrame.loc[:,'y'] = y


    return inputDataFrame# return inputDataframe with new columns


if __name__ == '__main__':
    data = loadData.loadRawData()
    window = data.iloc[0:1000,:]

    print (preprocess(window).head())

    print("kkm")