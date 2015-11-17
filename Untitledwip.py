import pandas as pd
import numpy as np
import scipy as sp
import numpy.linalg as npl
import scipy.stats as sps
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')


#Working directory (please enter your working directory)
os.chdir('/Users/duccioa/CLOUD/C07_UCL_SmartCities/QuantitativeMethods/wk6')
iris = pd.read_csv("iris.csv", sep = ",")
data = iris.ix[:, 0:4]
arr = iris.as_matrix()[:,0:4]





k = 4
nn = len(data.index)
data['group'] = g = np.random.randint(0, k, nn)#assign random cluster
centroids = data.groupby('group')
centroids = centroids.mean().as_matrix()#return a dataframe with each line is the multidimensional coords of a centroid


###FUNCTIONS###

#Multidimensional distance calculator
def distance_between(x1,x2):
    return npl.norm(x1-x2)
    
    
#return a matrix with the distance of each point from each centroid
#X is the matrix with the points (the data) and C is the matrix with the centroids
def dist_matrix(X, C):
    n = X.shape[0]
    m = C.shape[0]
    distances = np.zeros([n, m], dtype='float64')
    for i in range(0,n):
        for j in range(0,m):
            distances[i,j] = distance_between(X[i], C[j])
    return distances
    
    
#Assign cluster based on the minimum distance    
def assign_cluster(X):
    new_group = np.argmin(X, axis = 1)
    return new_group
    
#X is the data matrix, C is the first centroid matrix, g is the first grouping, num_it is the number of iteration
def clustering_iteration(X, C, g, num_it = 1000):
    n = X.shape[0]
    ng = np.zeros(n, dtype='int')
    while (not np.array_equal(g, ng)):
        g = ng
        D = dist_matrix(X, C)
        ng = assign_cluster(D)
    return ng
            

###CODE###
clusters = clustering_iteration(arr, centroids, g)
data['group'] = clusters

plt.figure(1)
plt.scatter(data['sepal_l'], data['sepal_w'], c = clusters, cmap = "Reds")
plt.figure(2)
plt.scatter(data['sepal_l'], data['petal_l'], c = clusters, cmap = "Blues")


plt.show(1)
plt.show(2)
