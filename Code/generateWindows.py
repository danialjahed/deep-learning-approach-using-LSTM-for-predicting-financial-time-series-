import os
import pandas as pd
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
                # self.windowAddress.append(directory)
            window.to_csv(directory+dataset_name+'.csv',index=None)
            return True , 'success'
        except:
            return False , 'error'
        
        
    def windowIndex(self):
        index = []
        maxSize = loadData.loadRawData(name= self.dataSetList[0]).shape[0] 
        print(maxSize)
        start = 0
        end = 0
        while(end != -1):
            end = start + self.defaultRange
            if end > maxSize:
                end = -1
            index.append((start,end))
            print(index)
            start = start + self.defaultOverlapRange
        
        return index


    def generateWindows(self):
        index = self.windowIndex()
        for dataset in self.dataSetList:
            for i in index:
                self.generateWindow(start_data=i[0],end_data=i[1],dataset_name=dataset)
        return True

    def loadData(self,windowList=[0],dataSetList=['NKE']):
        '''
            need to save this object as pickle to save the addresses 
            OR
            save addresses to a file and write another class for loading generated data

            * please load data for LSTM manually
        '''
        dfList = []
        for window in windowList:
            for dataset in dataSetList:
                dfList.append(loadData.loadRawDataByAddress(self.windowAddress[window]+dataset+'.csv'))
        return dfList

if __name__ == '__main__':
    w = Windowing()
    w.generateWindows()
    print(w.windowAddress)
    l = w.loadData()
    print(l[0].head())
    print('kkm')