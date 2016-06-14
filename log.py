# Implementation of two class classification using logistics regression
#TODO: make a python module containing the methods fit,predict,gradient descent and import that to use in regression module
#      which consists of linear and logit regression
#TODO: Non linear decision boundary using regularized logit regression

# X : no_of_samples x dimension ---------- 118 x 28 dimension mapped from 2 to 28 by feature mapping and bias term 
# Y : no_of_samples x 1------------------- 118 x1
# beta : dimension x 1 ------------------- 28 x1

# functions implemented
# fit()
# gradient descent : to learn the parameters beta by iterative method
# sgd : sstochastic gradient descent , a random mini batch of the data is used for parameter update
# plot2 : plot the non linear decision boundary , uses contour plot
# predict : calculate h(X,theta) and maps to 0 or 1
# feature_scale()
# map_features()


import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand

cost = np.zeros(30000)
def fit(X,Y):
	w = feature_scale(X)
	X = mapFeatures(X[:,1:X.shape[1]],6)
	no_of_variables = X.shape[1];	# no of columns
	#return gradient_descent(X,Y,no_of_variables,w);
	return sgd(X,Y,no_of_variables);
	
def gradient_descent(X,Y,n,w):
	no_of_iter = 30000
	alpha = 0.09
	m = Y.shape[0]
	beta = np.zeros(n)
	beta = np.matrix(beta)
	beta = beta.T
	while no_of_iter > 0:
		temp  = h(X,beta) 
		beta = beta - alpha*(X.T*(h(X,beta) - Y))/m
		cost[30000-no_of_iter] = (-Y.T*np.log(temp) - (1-Y.T)*np.log(1-temp))/m
		no_of_iter -= 1
	return beta

def h(X,beta):
	return np.reciprocal(1+np.exp(-(X*beta)))

# function to plot decision boundary for no_of_variables = 2
def plot(X,Y,beta):
	for i in range(1,X.shape[0]):
		if Y[i] == 1:
			plt.plot(X[i][1],X[i][2],'rs')
		else:
			plt.plot(X[i][1],X[i][2],'bs')
	pointX1 = beta[0]/beta[1]*-1
	pointX2 = beta[0]/beta[2]*-1 
	print pointX1,pointX2
	plt.plot([pointX1,0],[0,pointX2])
	plt.show()

def feature_scale(X):
	w = np.zeros((1,X.shape[1]));
	for i in range(1,X.shape[1]):
		maxim = np.max(X[:,i])
		minim = np.min(X[:,i])
		w[0][i] = maxim -minim
		mean = np.mean(X[:,1])
		X[:,i] -= mean
		X[:,i] /= w[0][i]
	return w

def predict(X,beta):
	temp = X*beta
	res = np.zeros(temp.shape[0])
	for i in range(0,temp.shape[0]):
		if temp[i] >= 0:
			res[i] = 1
		else:
			res[i] = 0
	return res
				
def mapFeatures(X1,degree):
	end = 0
	no_of_features = (degree+1)*(degree+2)/2; 
	out = np.ones((X1.shape[0],no_of_features))
	for i in range(0,degree+1):
		for j in range(0,i+1):
			out[:,end] = np.multiply(X1[:,0]**(i-j),X1[:,1]**(j))
			end +=1
	return out
	
def plot2(X,Y,beta):
	#function to plot non linear decison boundary
	for i in range(1,X.shape[0]):
		if Y[i] == 1:
			plt.plot(X[i][1],X[i][2],'rs')
		else:
			plt.plot(X[i][1],X[i][2],'bs')
	#Here is the grid range
	#50 values btw -1 and 1.5
    	u = np.linspace(-1, 1.5, 50);
    	v = np.linspace(-1, 1.5, 50);

    	z = np.zeros((len(u), len(v)));
    	#Evaluate z = theta*x over the grid
    	#decision boundary is given by theta*x = 0
    	for i in range(0,len(u)):
    		for j in range(1,len(v)):
    			temp = np.matrix([u[i],v[j]])
    	        	z[i,j] = mapFeatures(temp,6)*beta;
    	 
    	z = z.T; # important to transpose z before calling contour
	
	#Plot z = 0
	#Notice you need to specify the range [0, 0]
	plt.contour(u, v, z, [0, 0])
	
	plt.show()	

def sgd(X,Y,n):
	no_of_iter = 30000
	alpha = 0.1
	m = Y.shape[0]
	beta = np.zeros(n)
	beta = np.matrix(beta)
	beta = beta.T
	while no_of_iter > 0:
		temp  = h(X,beta)
		#print temp.shape
		r = rand.rand(40)*m;
		r = r.astype(int); 
		beta = beta - alpha*(np.matrix(X[r,:]).T*(temp[r,0] - Y[r,0]))/m
		cost[30000-no_of_iter] = (-Y.T*np.log(temp) - (1-Y.T)*np.log(1-temp))/m
		no_of_iter -= 1
	return beta
			
