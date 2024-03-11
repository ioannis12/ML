#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:39:15 2019

@author: ioannismilas
"""

import numpy as np
import matplotlib.pyplot as plt
from copulalib.copulalib  import Copula

x = np.random.normal(size = 100)
y = 2.5*x + np.random.normal(size = 100)

foo = Copula(x, y, family = 'clayton')

print(foo.tau)

print(foo.sr)

print(foo.pr)

print(foo.theta)

X1, Y1 = foo.generate_xy(1000)