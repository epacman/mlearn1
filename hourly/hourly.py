# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:40:01 2019

@author: Erik
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

filename = 'complete_snygg_fill.csv' 

yearcol = 0
monthcol = 1
daycol = 2
hourcol = 3
minutecol = 4
opencol = 5
highcol = 6
lowcol = 7
closecol = 8

row = 0
col = 0

rownum = 0
colnum = 0

hightemp = 0
lowtemp = 2000

temp = []
hour = []
minute = []
oopen = []
high = []
low = []
close = []


ifile = open(filename, 'rb')
reader = csv.reader(ifile)

#skapa array från  csv

for row in reader:
    if rownum == 0:
        header=row
    else:
        #print " hit!"
        for col in row:
            if colnum == hourcol:
                temp = float(col)
                print(temp)
                hour.append(temp)
            
            if colnum == minutecol:
                temp = float(col)
                minute.append(temp)
                
            if colnum == opencol:
                 temp = float(col)
                 oopen.append(temp)
                
            if colnum == highcol:
                temp = float(col)
            if colnum == lowcol:
                temp = float(col)
            if colnum == closecol:
                temp = float(col)

            colnum += 1
            print colnum
    rownum +=1
    
ifile.close()            
        
            