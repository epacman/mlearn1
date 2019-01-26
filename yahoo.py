# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:45:49 2019

@author: elindgre
"""

import pandas_datareader as pdr
import datetime

stocks = ["GPRO","TSLA"]
start = datetime.datetime(2012,5,31)
end = datetime.datetime(2018,3,1)

f = pdr.DataReader(stocks, 'yahoo',start,end)