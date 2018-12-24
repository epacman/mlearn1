# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:34:33 2018

@author: ELINDGRE
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
import numpy as np


def initialize(context):

    context.stocks = symbols('XLY',  # XLY Consumer Discrectionary SPDR Fund   
                           'XLF',  # XLF Financial SPDR Fund  
                           'XLK',  # XLK Technology SPDR Fund  
                           'XLE',  # XLE Energy SPDR Fund  
                           'XLV',  # XLV Health Care SPRD Fund  
                           'XLI',  # XLI Industrial SPDR Fund  
                           'XLP',  # XLP Consumer Staples SPDR Fund   
                           'XLB',  # XLB Materials SPDR Fund  
                           'XLU')  # XLU Utilities SPRD Fund
    
    context.historical_bars = 100
    context.feature_window = 10
    
def handle_data(context, data):
    prices = history(bar_count = context.historical_bars, frequency='1d', field='price')

    for stock in context.stocks:   
        ma1 = data[stock].mavg(50)
        ma2 = data[stock].mavg(200)
        
        start_bar = context.feature_window
        price_list = prices[stock].tolist()
        
        X = []
        y = []
        
        
while bar < len(price_list)-1:
            try:
                end_price = price_list[bar+1]
                begin_price = price_list[bar]
                
                pricing_list = []
                xx = 0
                for _ in range(context.feature_window):
                    price = price_list[bar-(context.feature_window-xx)]
                    pricing_list.append(price)
                    xx += 1
                    
                features = np.around(np.diff(pricing_list) / pricing_list[:-1] * 100.0, 1)
                
                if end_price > begin_price:
                    label = 1
                else:
                    label = -1

                bar += 1
                print(features)

            except Exception as e:
                bar += 1
                print(('feature creation',str(e)))