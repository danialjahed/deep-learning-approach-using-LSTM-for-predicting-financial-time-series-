import pandas as pd 
import numpy as np 
import scipy 
import os



def periodizeSignals(X,period):
    return X

def periodizePrices(X,period):
    return X


def calcReturn(predictedReturn,prices,currentPrice=None,period=1):
    Invesment = 1000
    fee = 0.1
    
    share = 0
    capital = Invesment
    
    seqDecisions = []
    
    periods =  periodizeSignals(predictedReturn,period)
    prices =  periodizePrices(prices,period)

    assert len(prices)==len(periods)
    print(len(prices))
    print(periods.head())
    for i in range(len(prices)):
        if periods[i] == 1:
            if share == 0:
                share = (capital-fee)/currentPrice
                capital = 0
                seqDecisions.append('buy')

            elif share != 0:
                seqDecisions.append('hold')

        elif periods[i] == 0:
            if share == 0:
                seqDecisions.append('hold')
            elif share != 0:
                capital = (share * currentPrice)-fee
                share = 0
                seqDecisions.append('sell')
        
        currentPrice = prices[i]
        # print(share,capital)
    
    if share == 0:
        return (capital,seqDecisions,capital/Invesment)
    elif share != 0:
        return (share * prices[len(prices)-1],seqDecisions,(share * prices[len(prices)-1])/Invesment)




if __name__=='__main__':
    predict = [1,0,1,1,0,1,0,0,0,1]
    price = [100,50,100,150,100,150,100,50,25,100]

    print(calcReturn(predict,price,50))