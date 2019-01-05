# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:45:38 2018

@author: ELINDGRE
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
#import pylab
#from mpl_toolkits.mplot3d import Axes3D

#Config
filename = 'allt.csv' 

lastcol = 8
hourcol = 3
minutecol = 4
daycol = 2

rownum = 0
colnum = 0

kurs = []
kurs_lunch = []
kurs_temp = 0
hour = []
hour_temp = 0
minute = []
minute_temp = []
day = []
day_temp = []
diff = []
updays = []
hour_data = []
hour_label = []

ifile = open(filename, 'rb')
reader = csv.reader(ifile)


#main loop, en csv-rad i taget
for row in reader:
	#om första raden, plocka ut text
    if rownum == 0:
        header=row
    else:
        colnum = 0
        for col in row:
            if colnum == lastcol:
                kurs_temp = float(col)
                kurs.append(kurs_temp)
               
            if colnum == daycol:
                day_temp = float(col)
                day.append(day_temp) 
                            
            if colnum == hourcol:
                hour_temp = float(col)
                hour.append(hour_temp)
                
            if colnum == minutecol:
                minute_temp = float(col)
                minute.append(minute_temp)
                
                
                
            colnum += 1
    rownum +=1
    
ifile.close()

i = 1
while  i < len(kurs):
    if hour[i] > hour[i-1]:
        hour_data.append(kurs[i])
        hour_label.append(hour[i])
    i+=1
    
import csv

with open('hourly.csv', 'wb') as csvfile:
    fieldnames = ['timme', 'kurs']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(hour_data)) :
        writer.writerow({'timme': hour_label[i], 'kurs': hour_data[i]})

    
