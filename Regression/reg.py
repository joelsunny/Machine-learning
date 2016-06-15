#TODO:X = m x (n+1) where m = no:of samples , n = no: of variables
#     Y = m x 1
#TODO:implement regression using normal eqn where it is advantageous over gradient descent
import numpy as np

# fit the dataset (X,Y) with a linear regression model and return the parameters of the model
# optimization algorithm used : simple gradient descent with fixed iterations
def fit(X,Y):
	# dimension of the multivariate regression problem
	no_of_variables = X.shape[1]
	return gradient_descent(X,Y,no_of_variables);
	
def gradient_descent(X,Y,n):
	no_of_iter = 1500
	alpha = 0.01
	m = Y.shape[0]
	beta = np.zeros(n)
	beta = np.matrix(beta)
	beta = beta.T
	while no_of_iter > 0:
		#beta = beta - alpha*(X.T*(X*beta-Y))/m
		beta = beta - alpha*(X.T*(h(X,beta) - Y))/m
		no_of_iter -= 1
	return beta
	
def h(X,beta):
	return X*beta
			
def predict(X,beta):
	print X
	X = np.matrix(X)
	x = np.ones((X.shape[0],X.shape[1]+1))
	x[:,1:X.shape[1]+1] = X
	print x
	return x*beta

