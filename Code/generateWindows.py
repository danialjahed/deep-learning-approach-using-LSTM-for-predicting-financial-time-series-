import os
import pandas as pd
import pickle as pk
import loadData
import preprocessing 

class Windowing:

    def __init__(self,baseAddress='../DataSets/windows/',dataSetList=['AAPL','INTC','KO','MSFT','NKE','ORCL','SNE','T','TM','VZ'],defaultRange=1000,defaultOverlapRange=250):
        self.baseAddress = baseAddress
        if not os.path.exists(self.baseAddress):
            os.makedirs(self.baseAddress)
        self.dataSetList = dataSetList
        self.windowAddress = []
        # self.dataSetAddress = []
        self.windowNumber = 0
        self.defaultRange = defaultRange
        self.defaultOverlapRange = defaultOverlapRange
    
    def __str__(self):
        return 'this is print test'

    
    def loadFromPickle(self,pkl):
        obj = pk.load(open(pkl,'rb'))
        # return obj
        self.baseAddress = obj.baseAddress
        self.dataSetList = obj.dataSetList
        self.windowAddress = obj.windowAddress
        # self.dataSetAddress = 
        self.windowNumber = obj.windowNumber
        self.defaultRange = obj.defaultRange
        self.defaultOverlapRange = obj.defaultOverlapRange
        del obj
        return True
        
    def saveToPickle(self,address=None):
        directory = self.baseAddress+"generateWindows"+".pickle"
        directory = open( directory, "wb" )
        pk.dump( self, directory )
        directory.close()
        return True

    def addDataSet_List(self,name):
        try:
            for i in name:
                self.dataSetList.append(i)
            return True
        except:
            return False



    def generateWindow(self,start_data,end_data,dataset_name):
        try:
            data = loadData.loadRawData(name=dataset_name)
            if start_data == -1:
                start_data = 0
            if end_data == -1:
                end_data = data.shape[0]-1        
            window = data.iloc[start_data:end_data,:]
            window = preprocessing.preprocess(window)
            directory = '{}windowFrom{}To{}/'.format(self.baseAddress,start_data,end_data)
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.windowAddress.append(directory)
            window.to_csv(directory+dataset_name+'.csv',index=None)
            return True , 'success'
        except:
            return False , 'error'
        
        
    def windowIndex(self):
        index = []
        maxSize = loadData.loadRawData(name= self.dataSetList[0]).shape[0] 
        # print(maxSize)
        start = 0
        end = 0
        while(end != -1):
            end = start + self.defaultRange
            if end > maxSize:
                end = -1
            index.append((start,end))
            self.windowNumber += 1
            # print(index)
            start = start + self.defaultOverlapRange
        
        return index


    def generateWindows(self):
        index = self.windowIndex()
        for dataset in self.dataSetList:
            for i in index:
                self.generateWindow(start_data=i[0],end_data=i[1],dataset_name=dataset)
        return True

    def loadData(self,windowList=[0],DSList=['NKE'],removeList=[]):
        temp = {}
        dfList = {}
        for window in windowList:
            for dataset in DSList:
                temp[dataset] = loadData.loadRawDataByAddress(self.windowAddress[window]+dataset+'.csv',removeList=removeList)
            dfList[str(window)] = temp
            temp = {}
        return dfList
    
    def printScheme(self):
        data = self.loadData(windowList=range(self.windowNumber),DSList=self.dataSetList)
        for WindowKey,Value in data.items():
            print(WindowKey)
            for Datasetkey,value in Value.items():
                print('\t'+Datasetkey)
        return True

if __name__ == '__main__':
    w = Windowing()
    w.generateWindows()
    w.saveToPickle()
    print('kkm')