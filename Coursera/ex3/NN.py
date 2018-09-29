# -*- coding: UTF-8 -*-
import scipy.io as scio
import random

data = scio.loadmat('ex3data1.mat')

# print type(data)
# print type(data['X'])
# print type(data['y'])
print random.sample(data['X'], 2)
