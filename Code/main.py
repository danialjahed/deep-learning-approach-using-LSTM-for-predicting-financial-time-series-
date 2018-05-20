import numpy as np 
import pandas as pd 
import scipy 
import tensorflow as tf 
import keras as k 
import pickle as pk
import os
import copy
import matplotlib.pyplot as plt
import seaborn

from generateWindows import Windowing 
from ModelEval import ModelEval 
from calcReturn import calcReturn

from DNN import model as DNNmodel
from randomForest import model as RFmodel
from logisticRegression import model as LRmodel

if __name__ == '__main__':


    pklAddress = '../DataSets/windows/generateWindows.pickle'
    windows = Windowing()
    windows.loadFromPickle(pklAddress)

    eModel = ModelEval(windowingModel=windows)
    eModel.loadFromPickle()
    
    # performance = []
    # objects = []
    # Return = []
    # for dataset in windows.dataSetList:
    #     eModel.loadOneInstanceOfData(windowNumber=0,datasetName=dataset)
    #     eModel.dataSplit()
    #     flag,model,trade,metrics = eModel.kerasEvalModel(DNNmodel)
    #     if flag:
    #         performance.append(metrics['acc']*100)
    #         objects.append(dataset)

    #         predictPrice = model.predict(trade.drop(columns=['y']))
    #         predictPrice = [predictPrice[i][0] for i in range(len(predictPrice))]
    #         predictPrice = pd.Series(predictPrice)
    #         predictPrice = predictPrice.drop(predictPrice.index[0])
    #         prices = trade['close']
    #         currentPrice = prices[0]
    #         prices = prices.drop(prices.index[0])
    #         predictPrice.index = range(len(predictPrice))
    #         prices.index = range(len(prices))
    #         capital,seqDecisions,rate = calcReturn(predictPrice,prices,currentPrice)
    #         print(seqDecisions)
    #         Return.append(rate)
    
    
    # y_pos = np.arange(len(objects))
    # plt.ylim(0,100)
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('performance')
    # plt.title('accuracy')
    # plt.show()

    # objects.append('average')
    # Return.append(sum(Return)/(len(objects)-1))
    # y_pos = np.arange(len(objects))
    # plt.ylim(-3,3)
    # plt.bar(y_pos, Return, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('rate')
    # plt.title('Return')
    # plt.show()
    
    

    def saveBarChartAcc(objects,performance,directory):
        y_pos = np.arange(len(objects))
        plt.ylim(0,100)
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('performance')
        plt.title('accuracy')
        plt.savefig(directory)


    def saveBarChartReturn(objects,Return,directory,range=-3):
        objects.append('average')
        Return.append(sum(Return)/(len(objects)-1))
        y_pos = np.arange(len(objects))
        plt.ylim(-3,3)
        plt.bar(y_pos, Return, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('rate')
        plt.title('Return')
        plt.savefig(directory)


    eModel.addSavedModel(DNNmodel,'DNN')
    eModel.addSavedModel(RFmodel,'RF')
    eModel.addSavedModel(LRmodel,'LR')
    # eModel.addSavedModel(,'LSTM')

    objects = windows.dataSetList
    baseDir = 'Result/'
    for slidingWindow in range(windows.windowNumber):
        DNNperformance = []
        RFperformance = []
        LRperformance = []
        # LSTMperformance = []
        
        DNNReturn = []
        RFReturn = []
        LRReturn = []
        # LSTMReturn = []

        directory = baseDir + 'window_' +str(slidingWindow)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for dataset in windows.dataSetList:
            eModel.loadOneInstanceOfData(windowNumber=slidingWindow,datasetName=dataset)
            eModel.dataSplit()
            flag,model,trade,metrics = eModel.EvalSavedModel(name='DNN',framework='keras')
            # print(slidingWindow,dataset)
            if flag:
                DNNperformance.append(metrics['acc']*100)
                # objects.append(dataset)

                predictPrice = model.predict(trade.drop(columns=['y']))
                predictPrice = [predictPrice[i][0] for i in range(len(predictPrice))]
                predictPrice = pd.Series(predictPrice)
                predictPrice = predictPrice.drop(predictPrice.index[0])
                prices = trade['close']
                currentPrice = prices[0]
                prices = prices.drop(prices.index[0])
                predictPrice.index = range(len(predictPrice))
                prices.index = range(len(prices))
                capital,seqDecisions,rate = calcReturn(predictPrice,prices,currentPrice)
                DNNReturn.append(rate)
                
            flag,model,trade,metrics = eModel.EvalSavedModel(name='RF',framework='sklearn')
            if flag:
                RFperformance.append(metrics['accuracy']*100)
                # objects.append(dataset)

                predictPrice = model.predict(trade.drop(columns=['y']))
                # print(predictPrice)
                # predictPrice = [predictPrice[i][0] for i in range(len(predictPrice))]
                predictPrice = pd.Series(predictPrice)
                predictPrice = predictPrice.drop(predictPrice.index[0])
                prices = trade['close']
                currentPrice = prices[0]
                prices = prices.drop(prices.index[0])
                predictPrice.index = range(len(predictPrice))
                prices.index = range(len(prices))
                capital,seqDecisions,rate = calcReturn(predictPrice,prices,currentPrice)
                RFReturn.append(rate)

            flag,model,trade,metrics = eModel.EvalSavedModel(name='LR',framework='sklearn')
            if flag:
                LRperformance.append(metrics['accuracy']*100)
                # objects.append(dataset)

                predictPrice = model.predict(trade.drop(columns=['y']))
                # predictPrice = [predictPrice[i][0] for i in range(len(predictPrice))]
                predictPrice = pd.Series(predictPrice)
                predictPrice = predictPrice.drop(predictPrice.index[0])
                prices = trade['close']
                currentPrice = prices[0]
                prices = prices.drop(prices.index[0])
                predictPrice.index = range(len(predictPrice))
                prices.index = range(len(prices))
                capital,seqDecisions,rate = calcReturn(predictPrice,prices,currentPrice)
                LRReturn.append(rate)

            # flag,model,trade,metrics = eModel.EvalSavedModel(name='LSTM',framework='keras')
            # if flag:
            #     LSTMperformance.append(metrics['acc']*100)
            #     # objects.append(dataset)

            #     predictPrice = model.predict(trade.drop(columns=['y']))
            #     predictPrice = [predictPrice[i][0] for i in range(len(predictPrice))]
            #     predictPrice = pd.Series(predictPrice)
            #     predictPrice = predictPrice.drop(predictPrice.index[0])
            #     prices = trade['close']
            #     currentPrice = prices[0]
            #     prices = prices.drop(prices.index[0])
            #     predictPrice.index = range(len(predictPrice))
            #     prices.index = range(len(prices))
            #     capital,seqDecisions,rate = calcReturn(predictPrice,prices,currentPrice)
            #     LSTMReturn.append(rate)
        
        saveBarChartAcc(objects,DNNperformance,directory+'DNNperformance.png')
        saveBarChartAcc(objects,RFperformance,directory+'RFperformance.png')
        saveBarChartAcc(objects,LRperformance,directory+'LRperformance.png')
        # saveBarChartAcc(objects,LSTMperformance,directory+'LSTMperformance.png')

        saveBarChartReturn(copy.deepcopy(objects),copy.deepcopy(DNNReturn),directory+'DNNreturn.png')
        saveBarChartReturn(copy.deepcopy(objects),copy.deepcopy(RFReturn),directory+'RFreturn.png')
        saveBarChartReturn(copy.deepcopy(objects),copy.deepcopy(LRReturn),directory+'LRreturn.png')
        # saveBarChartReturn(copy.deepcopy(objects),LSTMReturn,directory+'LSTMreturn.png')