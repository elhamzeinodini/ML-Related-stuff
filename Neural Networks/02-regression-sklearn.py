import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as li
import sklearn.metrics as met

np.random.seed(0)
plt.style.use('ggplot')

dataLength = 1000
featureCount = 3


X = np.zeros((dataLength, featureCount))

Y = np.zeros((dataLength, 1))

for i in range(dataLength) :
    X[i, :] = np.random.uniform(-1, 1, featureCount)
    Y[i, 0] = 1.2*X[i,0] - 0.9*X[i,1] + 0.4*X[i,2] - 0.5
    

Model = li.LinearRegression()
Model.fit(X, Y)

modelResult = Model.predict(X)

MSE = met.mean_squared_error(Y, modelResult)
MAE = met.mean_absolute_error(Y, modelResult)
R2S = met.r2_score(Y, modelResult)

print(f'MSE : {round(MSE, 4)}')
print(f'MAE : {round(MAE, 4)}')
print(f'R2S : {round(R2S, 4)}')

print(f'coefficients : {Model.coef_}')
print(f'bias: {Model.intercept_}')

plt.scatter(Y[:, 0], modelResult[:, 0], s=12, c="teal")
plt.plot([-3, 2], [-3, 2], lw=1.2, c="crimson", label="x=y")
plt.title("Regression Plot")
plt.xlabel("Target Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()