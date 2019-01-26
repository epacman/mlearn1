# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:18:46 2019

@author: elindgre
"""

import numpy

for i in range(len(first_minute)):
    numpy.savetxt("foo.csv", first_minute[i], delimiter=",")