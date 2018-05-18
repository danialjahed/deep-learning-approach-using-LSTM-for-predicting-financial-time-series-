import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score,confusion_matrix
from generateWindows import Windowing 
import pickle as pk
import os

class ModelEval:

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

    def loadFromPickle(self,name='sklearnModelEval'):

        directory = self.baseAddress + name + '.pickle'
        # print(directory)
        obj = pk.load(open(directory,'rb'))
        # print(obj.instanceData)
        self.baseAddress =  obj.baseAddress
        # self.windowingPKLAddress = windowingPKLAddress
        self.EvalModelName =  obj.EvalModelName
        self.windowingModel =   obj.windowingModel
        self.featureRemoveList =  obj.featureRemoveList
        self.data =  obj.data
        self.instanceData =  obj.instanceData
        self.learningData =  obj.learningData
        self.Trade =  obj.Trade
        self.X =  obj.X
        self.Y =   obj.Y
        self.X_train =  obj.X_train
        self.X_test =  obj.X_test
        self.Y_train =  obj.Y_train
        self.Y_test =  obj.Y_test
        self.testSize =  obj.testSize
        self.savedModel =  obj.savedModel
        self.savedTrainedModel =  obj.savedTrainedModel
        self.metrics =  obj.metrics
        return True

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
    
        self.data  = self.windowingModel.loadData(windowList=range(self.windowingModel.windowNumber),DSList=self.windowingModel.dataSetList,removeList=self.featureRemoveList)
        return True

    def loadOneInstanceOfData(self,windowNumber=0, datasetName='NKE' ):
        try:
            windowNumber = str(windowNumber)
            self.instanceData = self.data[windowNumber][datasetName]
            return True
        except:
            return False

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

    def sklearnEvalModel(self,model=None,name=None,doItSaveModel=False,doItSaveModelResult=False):
        # try:
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
            if doItSaveModelResult:
                if name != None:
                    self.savedTrainedModel.append({name:{'type':'keras', 'model':model,'metrics':self.metrics}})
                else:
                    return (False, 'name is empty in eval model')
            
            return (True,model,self.Trade,self.metrics)
        
        else:
            return (False, 'Model is empty in eval model')
        
        # except expression as identifier:
        #     return (False,'Error occured in evalModel',identifier)

    def kerasEvalModel(self,model=None,name=None,doItSaveModel=False,doItSaveModelResult=False,Train_batch_size=15,Test_batch_size=100, epochs=100):
        try:
            if model != None:
                model.fit(self.X_train,self.Y_train,batch_size=Train_batch_size, epochs=epochs)
                prediction = model.evaluate(self.X_test, self.Y_test, batch_size=Test_batch_size)
                #################fill metrics####################################################
                # for (i,j) in (model.metrics_names,prediction):
                #     self.metrics[i] = j
                
                print(self.metrics)
                if doItSaveModel:
                    if name != None:
                        self.savedModel[name] = model
                    else:
                        return (False, 'name is empty in eval model')
                if doItSaveModelResult:
                    if name != None:
                        self.savedTrainedModel.append({name:{'type':'keras', 'model':model,'metrics':self.metrics}})
                    else:
                        return (False, 'name is empty in eval model')
                
                return (True,model,self.Trade,self.metrics)
            
            else:
                return (False, 'Model is empty in eval model')
        
        except:
            return (False,'Error occured in evalModel')

    def EvalSavedModel(self,name=None,framework='sklearn'):
        try:
            if name == None:
                return (False,'did not pass the name')

            if name in self.savedModel:
                if framework == 'sklearn':
                    return self.sklearnEvalModel()
                elif framework == 'keras':
                    return self.kerasEvalModel()
                else:
                    print(False,"framework error")    
            else:
                return (False,'model does not exist')
        except:
            return (False,'probably model did not save')

    def initialize(self,pklname='sklearnModelEval',windowNumber=0,datasetName='NKE'):
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




if __name__ == '__main__':
    #***************using func below that is commented*****************#
    
    # print('randomforest:')
    # sklernModelEval(ModelName='randomforest')
    # print('logisticregression:')
    # sklernModelEval(ModelName='logisticregression')
    
    #***************using class*****************#
    
    ##
    pklAddress = '../DataSets/windows/generateWindows.pickle'
    windows = Windowing()
    windows.loadFromPickle(pklAddress)
    eModel = ModelEval(windowingModel=windows)
    eModel.loadWindowsData()
    # print(eModel.data)
    eModel.loadOneInstanceOfData()
    eModel.dataSplit()
    # print(eModel.instanceData,"asdsajdajdagds")
    eModel.saveToPickle()
    ##

    ###
    # eModel = ModelEval(windowingModel=windows)
    # eModel.loadFromPickle()
    # # print(eModel.instanceData)
    ###
    # try:
    #     pass
    # except expression as identifier:
    #     pass
    ####
    # print(eModel.instanceData)
    # s = eModel.sklearnEvalModel(model= RandomForestClassifier(max_depth=20,n_estimators=1000,random_state=42))
    # print(s)
    ####



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



