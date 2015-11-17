import pandas as pd
import numpy as np
import scipy as sp
import numpy.linalg as npl
import scipy.stats as sps
import matplotlib.pyplot as plt
import os

#Working directory (please enter your working directory)
os.chdir('/Users/duccioa/CLOUD/C07_UCL_SmartCities/QuantitativeMethods/wk6')
iris = pd.read_csv("iris.csv", sep = ",")
data = iris.ix[:, 0:4]


def centroid(data, group_by):
    data.groupby(group_by)
    centroids = data.mean()
    

def k_mean(data, k):
    n = data[0].size()
    data['group'] = np.random.randint(0, k, n)#assign random cluster
    
    

