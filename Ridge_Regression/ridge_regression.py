import numpy as np
import matplotlib.pyplot as plt






def apply_linear_regression(data):

	X_train=data[:,0][:80]
	Y_train=data[:,1][:80]

	
	

	alpha=0.001
	theta=np.array([1,1])
	iterations=100
	X_1=X_train

	for i in range(iterations):
		theta_0=theta[0]- alpha * (1/len(Y_train)) * np.sum([np.dot(X_train[i],theta)- Y_train[i] for i in range(len(Y_train))])
		theta_1=theta[1] - alpha * (1/len(Y_train)) * np.sum([np.dot(X_1[i],np.dot(X_train[i],theta)-Y_train[i])for i in range(len(Y_train))])
		theta= np.array([theta_0,theta_1])
		

	
	return theta

def mse(data,prediction):
	line=prediction[1]*data[:,0] + prediction[0]
	true_y=data[:,1]
	square_error=[]
	for i in range(len(line)):
		square_error.append((line[i]-true_y[i])**2)
	return sum(square_error)/len(line)




data=np.load("linRegData.npy")
l=apply_linear_regression(data)

MSE=mse(data,l)
print(MSE)

X_test=data[:,0][80:100]
Y_test=data[:,1][80:100]
line=l[1]*X_test + l[0]
line2=l[1]*data[:,0] + l[0]




# Plot outputs
plt.figure(1)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot('231')
plt.scatter(data[:,0],data[:,1],  color='black')
plt.title("plot_of_data")

plt.subplot('232')
plt.scatter(data[:,0],data[:,1],  color='black')
plt.plot(data[:,0],line2, color='blue')
plt.title("regression_line_fullData")

plt.subplot('233')
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, line, color='blue')
plt.title("regressionline_testData")

plt.show()


