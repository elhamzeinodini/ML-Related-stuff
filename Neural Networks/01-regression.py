import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# this line is like setting a rule for picking random numbers, in this case the rule is 
# to always start as the specific position 0
# as a result, you'll always pull out the same sequence of numbers and the results are 
# re-producible
np.random.seed(0)
plt.style.use('ggplot')

dataLength = 1000
featureCount = 3

# np.zeros creates a new array, fills the entire array with zeros
# with the provided arguments, it creates a 2d array with 1000 rows and 3 columns (1000*3)
# it creates a matrix in this case
X = np.zeros((dataLength, featureCount))

# we only have 1 feature here (1000*1)
Y = np.zeros((dataLength, 1))

# going from 0 to 999
for i in range(dataLength) :
    # selecting row i of matrix X and all columns in that row
    # generating 3 random values between -1 (inclusive) and 1(exclusive) and populating all columns of row i 
    # of matrix X
    X[i, :] = np.random.uniform(-1, 1, featureCount)
    # noise
    e = np.random.uniform(-1, 1)
    # coefficients are preferred
    # populating row i and column 0 of matrix Y (matrix Y has one column only)
    Y[i, 0] = 1.2*X[i,0] - 0.9*X[i,1] + 0.4*X[i,2] - 0.5 + e
    

# -1 is inclusive : any random number can be -1 
# 1 is exclusive : no random number can be 1, the range goes up to 1 , but doesn't touch 1

# for i in range(featureCount):
#     plt.scatter(X[:, i], Y[:, 0], s=12)
#     plt.xlabel(f'X{i+1}')
#     plt.ylabel('Y')
#     plt.show()
    
    
def Model (A: np.ndarray, B: float, X: np.ndarray) :
    # A: 3*1 , B:  A floating-point number representing a bias term , X: 1000*3
    res = np.dot(X,A) + B
    # np.dot returns a matrix, res is : 1000 * 1
    return res.reshape((-1, 1))


# why do we need bias ? 

# Imagine you're trying to predict how much popcorn you'll need for a movie night based on 
# the number of people attending. You figure a linear relationship might work: more people, more popcorn.

# Linear Model:  This is like drawing a straight line on a graph, where the x-axis represents the number of 
# people and the y-axis represents the amount of popcorn.

# Perfect Fit (not realistic):  In a perfect world, all the data points (number of people, amount of 
# popcorn they eat) would fall exactly on this line.

# Real-world Data:  People have different appetites, some might share popcorn, and there might be other 
# snacks. So, the data points will likely scatter around the line, not forming a perfect fit.

# Bias as an Adjustment:  Imagine you're drawing a straight line to predict something, like movie 
# night popcorn based on the number of people. The bias term (B) is like a magic button that lets 
# you move the whole line up or down. This helps you get the line closer to the actual popcorn needs 
# of your friends (data points), even if some eat more or less than average

# There are two main reasons why you might need to move the line up or down in linear regression using the bias term (B):
# Linear models are good at capturing linear trends, but the real world is rarely perfectly linear


# Upward Bias (B positive): If your model consistently underestimates (predicts lower values than reality),
# you might need to move the line up with a positive bias. This is like adding a constant "buffer" to your 
# predictions to account for missing factors or data variations.

# Downward Bias (B negative): In the opposite case, if your model overestimates (predicts higher values),
# you might use a negative bias to shift the line down.

# res.reshape(2,3) creates a 2D array with 2 rows and 3 columns 
# res.reshape(-1,1) creates an array where we have as many rows as possible while preserving the total 
# number of the elements from the original array and 1 column


def Error(P: np.ndarray, X: np.ndarray, Y: np.ndarray) :
    A = P[:-1]
    B = P[-1]
    res = Model(A, B, X)
    # ei = Y - res , real minus model predicted value , to the power of 2 
    # using root mean squared error 
    error = np.mean(np.power(Y - res, 2))
    print(f'MSE: {error}')
    return error


# as in regression formula: y^ = b + a1x1 + a2x2 + ... + afxf
parametersCount = featureCount + 1

# P0 is the starting point for the optimization algorithm, it is an array
P0 = np.random.uniform(-1, 1, parametersCount)

# args(X, Y) are additional arguments that will be passed to the Error function during optimization.
# slsqp : optimization algorithm
result = opt.minimize(Error, P0, args = (X, Y), method='slsqp')
print(result)

finalResult = result['x']
print(finalResult)

A = finalResult[:-1]
B = finalResult[-1]

modelResult = Model(A, B, X)

plt.scatter(Y[:, 0], modelResult[:, 0], s=12, c="teal")
# drawing a red line
plt.plot([-3, 2], [-3, 2], lw=1.2, c="crimson", label="x=y")
plt.title("Regression Plot")
plt.xlabel("Target Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# data is without noise , go up and add a noise using "e = np.random.uniform(-1, 1)"