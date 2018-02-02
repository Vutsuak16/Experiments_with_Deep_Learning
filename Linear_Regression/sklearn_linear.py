import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score


data=np.load("linRegData.npy")

X_train=data[:,0][:80].reshape(-1,1)
Y_train=data[:,1][:80].reshape(-1,1)

X_test=data[:,0][80:100].reshape(-1,1)
Y_test=data[:,1][80:100].reshape(-1,1)

regression = linear_model.LinearRegression()

regression.fit(X_train,Y_train)

prediction = regression.predict(X_test)

x=data[:,0].reshape(-1,1)
line=regression.coef_*x + regression.intercept_ 



# Plot outputs
plt.figure(1)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot('211')
plt.scatter(data[:,0].reshape(-1,1),data[:,1].reshape(-1,1),  color='black')
plt.plot(x,line, color='blue')
plt.title("regressionline_sklearn_fullData")

plt.subplot('212')
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, prediction, color='blue')
plt.title("regressionline_sklearn_TestData")



plt.show()


print('Coefficient: \n', regression.coef_)
print('Intercept: \n', regression.intercept_)

print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, prediction))

print('Variance score: %.2f' % r2_score(Y_test, prediction))


'''
Coefficient: 
 [[ 0.36046113]]
Intercept: 
 [ 0.71784123]
Mean squared error: 0.62
Variance score: -16.53'''