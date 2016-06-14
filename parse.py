import numpy as np

n = 2
data = np.loadtxt("ex2data2.txt",delimiter = ",")
data = np.matrix(data)
x = data[:,0:n]
y = data[:,n]	# last column
X = np.ones((x.shape[0],n+1))
X[:,1:n+1] = x
