import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score,confusion_matrix
from generateWindows import Windowing 
import pickle as pk
import os

class sklearnModelEval:
    
    def __init__(self, baseAddress='Modelevl/',windowingModel=None,evalModelName='sklearnModelEval',featureRemoveList=['date','adjclose','return','Sreturn'], testSize=0.2):
        
        if windowingModel == None:
            raise ValueError('windowModel did not pass')
        elif windowingModel.windowAddress == []:
            raise ValueError('windowModel is empty Model ')


        self.baseAddress = baseAddress
        if not os.path.exists(self.baseAddress):
            os.makedirs(self.baseAddress)
        # self.windowingPKLAddress = windowingPKLAddress
        self.EvalModelName = evalModelName
        self.windowingModel = windowingModel
        self.featureRemoveList = featureRemoveList
        self.data = None
        self.instanceData = None
        self.learningData = None 
        self.Trade = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.testSize = testSize
        self.savedModel = {} 
        self.savedTrainedModel = []
        self.metrics = {} 
    
    def loadWindowingModelFromPickle(self,name='sklearnModelEval'):
        try:
            directory = self.baseAddress + name + '.pickle'
            obj = pk.load(open(directory,'rb'))
            self = obj
            return True
        except:
            return False
    
    def saveToPickle(self):
        try:
            directory = self.baseAddress + self.EvalModelName + '.pickle'
            directory = open( directory, "wb" )
            pk.dump( self, directory )
            directory.close()
            return True
        except:
            return False

    def loadWindowsData(self):
        try:
            self.data  = self.windowingModel.loadData(windowList=range(self.windowingModel.windowNumber),DSList=self.windowingModel.dataSetList,removeList=self.featureRemoveList)
            return True
        except:
            return False

    def loadOneInstanceOfData(self,windowNumber=0, datasetName='NKE' ):
        windowNumber = str(windowNumber)
        return self.data[windowNumber][datasetName]

    def dataSplit(self):
        try:   
            self.learningData = self.instanceData.iloc[0:self.windowingModel.defaultRange-self.windowingModel.defaultOverlapRange,:]
            self.Y = self.learningData.y
            self.X = self.learningData.drop(columns=['y'])
            self.Trade =  self.instanceData.iloc[self.windowingModel.defaultRange-self.windowingModel.defaultOverlapRange:self.windowingModel.defaultRange,:]
            self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y,test_size=self.testSize)
        except:
            return False

    def addSavedModel(self,Model,name):
        try:
            self.savedModel[name] = Model
            return True
        except:
            return False        

    def getSavedModel(self):
        list = []
        for key,_ in self.savedModel.items():
            print(key)
            list.append(key)
        return list

    def evalModel(self,model=None,name=None,doItSaveModel=False,doItSaveModelResult=False):
        try:
            if model != None:
                model.fit(self.X_train,self.Y_train)
                prediction = model.predict(self.X_test)
                self.metrics['accuracy'] = accuracy_score(self.Y_test,prediction)
                self.metrics['precision'] = precision_score(self.Y_test,prediction)
                self.metrics['recall'] = recall_score(self.Y_test,prediction)
                self.metrics['PRcurve'] = precision_recall_curve(self.Y_test,prediction)
                self.metrics['f1'] = f1_score(self.Y_test,prediction)
                self.metrics['confusion'] = confusion_matrix(self.Y_test,prediction)
                print('#acc:{} #prec:{} #recall:{} #f1:{}'.format(self.metrics['accuracy'],self.metrics['precision'],self.metrics['recall'],self.metrics['f1']))
                print(self.metrics['confusion'])
                if doItSaveModel:
                    if name != None:
                        self.savedModel[name] = model
                    else:
                        return (False, 'name is empty in eval model')
                if doItSaveModel:
                    if name != None:
                        self.savedTrainedModel.append({name:{'model':model,'metrics':self.metrics}})
                    else:
                        return (False, 'name is empty in eval model')
                
                return (True,model,self.Trade,self.metrics)
            
            else:
                return (False, 'Model is empty in eval model')
        
        except:
            return (False,'Error occured in evalModel')

    def evalSavedModel(self,name=None):
        try:
            if name == None:
                return (False,'did not pass the name')

            if name in self.savedModel:
                return self.evalModel()
            else:
                return (False,'model does not exist')
        except:
            return (False,'probably model did not save')

    def initialize(self,pklname='sklearnModelEval',windowNumber=0,datasetName='NKE'):
        if self.loadWindowingModelFromPickle(name=pklname):
            if self.loadWindowsData():
                if self.loadOneInstanceOfData(windowNumber=windowNumber, datasetName=datasetName):
                    if self.dataSplit():
                        return (True,'initialization completed')
                    else:
                        return (False, 'data split had error')
                else:
                    return (False,'not load on instance of data')
            else:
                return (False,'not load windows data')   
        else:
            return (False,'not load pickle')        


# def sklernModelEval(Model=None, ModelName=''):
#     pklAddress = '../DataSets/windows/generateWindows.pickle'
#     windows = Windowing()
#     if windows.loadFromPickle(pklAddress):
#         # print('data loader is ready')
#         removeList = ['date','adjclose','return','Sreturn']
#         data  = windows.loadData(windowList=range(windows.windowNumber),DSList=windows.dataSetList,removeList=removeList)
#         # windows.printScheme()
#         data = data['0']['NKE']
#         learningData = data.iloc[0:windows.defaultRange-windows.defaultOverlapRange,:]
#         Y = learningData.y
#         X = learningData.drop(columns=['y'])
#         Trade =  data.iloc[windows.defaultRange-windows.defaultOverlapRange:windows.defaultRange,:]
#         X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25) 
        
#         if Model != None:
#             model = Model
#         elif ModelName.lower() == 'randomforest':
#             model = RandomForestClassifier(max_depth=20,n_estimators=1000,random_state=42)
        
#         elif ModelName.lower() == 'logisticregression':
#             model = LogisticRegression()
#         else:
#             return (False,'model not set correctly')
#         model.fit(X_train,Y_train)
#         prediction = model.predict(X_test)
#         metrics = {}
#         metrics['accuracy'] = accuracy_score(Y_test,prediction)
#         metrics['precision'] = precision_score(Y_test,prediction)
#         metrics['recall'] = recall_score(Y_test,prediction)
#         metrics['PRcurve'] = precision_recall_curve(Y_test,prediction)
#         metrics['f1'] = f1_score(Y_test,prediction)
#         metrics['confusion'] = confusion_matrix(Y_test,prediction)
#         print('#acc:{} #prec:{} #recall:{} #f1:{}'.format(metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1']))
#         print(metrics['confusion'])
#         return (True,model,Trade,metrics)
#     else:
#         return (False,'window generator not loaded correctly')


if __name__ == '__main__':
    # print('randomforest:')
    # sklernModelEval(ModelName='randomforest')
    # print('logisticregression:')
    # sklernModelEval(ModelName='logisticregression')
    pass