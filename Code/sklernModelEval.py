import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score,confusion_matrix
from generateWindows import Windowing 
import pickle as pk


def sklernModelEval(Model=None, ModelName=''):
    pklAddress = '../DataSets/windows/generateWindows.pickle'
    windows = Windowing()
    if windows.loadFromPickle(pklAddress):
        # print('data loader is ready')
        removeList = ['date','adjclose','return','Sreturn']
        data  = windows.loadData(windowList=range(windows.windowNumber),DSList=windows.dataSetList,removeList=removeList)
        # windows.printScheme()
        data = data['0']['NKE']
        learningData = data.iloc[0:windows.defaultRange-windows.defaultOverlapRange,:]
        Y = learningData.y
        X = learningData.drop(columns=['y'])
        Trade =  data.iloc[windows.defaultRange-windows.defaultOverlapRange:windows.defaultRange,:]
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25) 
        
        if Model != None:
            model = Model
        elif ModelName.lower() == 'randomforest':
            model = RandomForestClassifier(max_depth=20,n_estimators=1000,random_state=42)
        
        elif ModelName.lower() == 'logisticregression':
            model = LogisticRegression()
        else:
            return (False,'model not set correctly')
        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        metrics = {}
        metrics['accuracy'] = accuracy_score(Y_test,prediction)
        metrics['precision'] = precision_score(Y_test,prediction)
        metrics['recall'] = recall_score(Y_test,prediction)
        metrics['PRcurve'] = precision_recall_curve(Y_test,prediction)
        metrics['f1'] = f1_score(Y_test,prediction)
        metrics['confusion'] = confusion_matrix(Y_test,prediction)
        print('#acc:{} #prec:{} #recall:{} #f1:{}'.format(metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1']))
        print(metrics['confusion'])
        return (True,model,Trade,metrics)
    else:
        return (False,'window generator not loaded correctly')


if __name__ == '__main__':
    print('randomforest:')
    sklernModelEval(ModelName='randomforest')
    print('logisticregression:')
    sklernModelEval(ModelName='logisticregression')