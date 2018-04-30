import pandas as pd
import numpy as np

def loadRawData(name='NKE'):
    raw_data = pd.read_csv('../DataSets/'+name+'.csv')
    raw_data = raw_data.rename(index=str, columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjclose","Volume":"volume"})
    return raw_data

def loadRawDataByAddress(address):
    raw_data = pd.read_csv(address)
    raw_data = raw_data.rename(index=str, columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjclose","Volume":"volume"})
    return raw_data

def adjustData(name='NKE'):
    data = loadRawData(name)
    close = data['close']
    adjustedClose = data['adjclose']
    adjustRatio = close/adjustedClose

    data['open'] = data['open']/adjustRatio
    data['close'] = data['close']/adjustRatio
    data['high'] = data['high']/adjustRatio
    data['low'] = data['low']/adjustRatio

    data.to_csv("../DataSets/Adjusted_"+name+".csv",index=False)

def loadAdjustedData(name='NKE'):
    adjData = pd.read_csv('../DataSets/Adjusted_'+name+'.csv')
    return adjData

def adjustAndLoad(name='NKE'):
    adjustData(name)
    return loadAdjustedData(name)

def loadAdjustedDevelopingData(name='NKE' , percent=10):
    if percent<=0 or percent>100:
        percent = 100
    adjData = loadAdjustedData(name)
    idx = int(np.ceil((len(adjData)*percent)/100))
    subsetData = adjData.iloc[0:idx,:]
    return subsetData


if __name__ == '__main__':
    print("this is phase 0")
