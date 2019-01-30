# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:57:56 2019

@author: elindgre
"""

#kurs: Varje minut
#minute: vilken minut som rådde
#hour: vilken timme som rådde
#day: vilken dag som rådde

import csv
import matplotlib.pyplot as plt
import numpy as np


hour_change = []
day_change = []
candlestick = ["open", "high", "low", "close", "is_first_hour", "is_last_hour"]

for i in range(len(kurs)):
    if hour[i] < hour[i-1]:
        day_change.append(1)
    else:
        day_change.append(0)
    
    if minute[i] < minute[i-1]:
        hour_change.append(1)
    else:
        hour_change.append(0)
        
#loopa igenom allt igen, leta min och max tills timme byts, lägg till candlestick i ny vektor

hourmax = 0;
hourmin = 5000;
high = []
low = []
oopen = []
close = []
is_last_hour = []

for j in range(len(kurs)-1):
    if j > 0 and j < len(kurs):
        if day_change[j] == 0:
            is_last_hour.append(0)
        else:
            is_last_hour.append(1)
    
        if hour_change[j] == 0:
            hourmax = max(hourmax,kurs[j])
            hourmin = min(hourmin,kurs[j])
            
        else:
            #spara data från timmen som varit
            high.append(hourmax)
            low.append(hourmin)
            #close[3] och open[3] kommer inte att avse samma timme. Fixa!
            close.append(kurs[j-1])
            oopen.append(kurs[j])      
            hourmax = 0
            hourmin = 5000

high=high[1:-1]
low=low[1:-1]
close = close[1:-1]

#allt är bra
#kolla
plt.plot(kurs[0:120])

candles = np.zeros((2,5))
hours_set = []

#for k in range(len(oopen)+1):
#    if k > 1:
#        hours_set.append(oopen[k])
#        hours_set.append(low[k])
#        hours_set.append(high[k])
#        hours_set.append(close[k])
#        hours_set.append(is_last_hour[k])
#        
        
#sätt ihop till stor array först, sedan stitcha ihop lagom många rader till feature-set
        
#candle = np.zeros(5)        
for k in range(len(oopen)-2):
    candle = [oopen[k],low[k],high[k],close[k],is_last_hour[k]]
#    np.append(candles,candle,axis=0)
    print candle
    
    
            
            