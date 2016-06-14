# Implementation of K-means clustering algorithm
# fit(X,K) :- dataset X and number of clusters K

import random
import numpy as np
import matplotlib.pyplot as plt

def fit(X,K):
      u = np.zeros((K,X.shape[1]))
      rand_init(X,u,K)
      print u
      u = np.matrix(u)
      converge = False
      label = np.zeros((X.shape[0],1))
      iter = 10
      i = 0
      while i < iter:
          label = assign(X,u,K)
          prev_u = u
          u = centroid(X,label,K)
          plot(X,label,u)
          input("pgm paused")
          i += 1

def rand_init(X,u,K):
      for i in range (0,K):
          u[i,:] = X[random.randint(0,X.shape[0]-1),:]

def assign(X,u,K):
      min_dist = 0
      temp = 0	
      label = np.zeros((X.shape[0],1))
      for i in range (0,X.shape[0]):
          min_dist = np.sum(np.power(X[i,:]-u[0,:],2));
          label[i][0] = 0
          for j in range(1,K):
              temp = np.sum(np.power(X[i,:] - u[j,:],2)) 
              if temp < min_dist:
                  min_dist = temp
                  label[i][0] = j
      return label

def centroid(X,label,K):
      temp = np.zeros((K,X.shape[1]))
      temp = np.matrix(temp)
      count = np.zeros(K)
      for i in range (0,X.shape[0]):
          temp[int(label[i][0]),:] += X[i,:]
          count[int(label[i][0])] += 1
      for i in range(0,K):
          if count[i] == 0:
              count[i] = 1
          temp[i,:] /= count[i]
      return temp

def plot(X,label,u):
      print label
      colors = ['b+','r+','g+']
      for i in range(0,X.shape[0]):
          plt.plot(X[i,0],X[i,1],colors[int(label[i,0])])
      for i in range(0,3):
      	  plt.plot(u[i,0],u[i,1],'rs')
      plt.show()

