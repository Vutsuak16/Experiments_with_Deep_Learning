import numpy as np
import matplotlib.pyplot as plt






def apply_ridge_regression(data,ridge_lambda):

	X_train=data[:,0][:80]
	Y_train=data[:,1][:80]

	
	

	alpha=0.00000000000000001
	theta=np.array([1]*16)
	iterations=100
	X_1=X_train

	for k in range(iterations):
		for j in range(16):
			#theta_0=theta[0]- alpha * (1/len(Y_train)) * np.sum([np.dot(X_train[i],theta)- Y_train[i] for i in range(len(Y_train))])
			theta[j]=theta[j]*(2*ridge_lambda) - alpha * (1/len(Y_train)) * np.sum([np.dot(np.power(X_1[i],j),np.dot(np.power(X_train[i],j),theta)-Y_train[i])for i in range(len(Y_train))])
		
		

	
	return theta






data=np.load("linRegData.npy")
ridge_lambdas=[0.01,0.05,0.1,0.5,1,5,10]
thetas=[]
for a in ridge_lambdas:
	l=apply_ridge_regression(data,a)
	thetas.append(l)

ct=0
avg_cross_val=[]

for i in thetas:
	ct=0
	error=[]
	while True:
		np.random.shuffle(data)
		val_data_X=data[:,0][ct:20+ct]
		val_data_Y=data[:,1][ct:20+ct]
		error_sum=0
		for j in range(20):
			y=0
			for k in range(16):
				y+=i[k]*(val_data_X[j]**k)
			error_sum+=(val_data_Y[j])-y
		error.append(error_sum)
		ct+=20
		if ct==100:
			avg_cross_val.append((sum(error)/20))
			break

index=avg_cross_val.index(min(avg_cross_val))


test_error=[]
for i in thetas:
	
	np.random.shuffle(data)
	test_data_X=data[:,0][80:]
	test_data_Y=data[:,1][80:]
	error_sum=0
	for j in range(20):
		y=0
		for k in range(16):
			y+=i[k]*(test_data_X[j]**k)
		error_sum+=(test_data_Y[j])-y
	test_error.append(error_sum/20)



plt.figure(1)


plt.subplot('231')
plt.xlabel("Lambdas")
plt.ylabel("cross_validation_error")
axes = plt.gca()
axes.set_ylim([-2,2])
plt.scatter(ridge_lambdas,avg_cross_val, color='black')
plt.title("cross_validation_error vs lambdas")

plt.subplot('232')
plt.xlabel("Lambdas")
plt.ylabel("test_error")
axes = plt.gca()
axes.set_ylim([-2,2])
plt.scatter(ridge_lambdas,test_error, color='red')
plt.title("test_error vs lambdas")

plt.subplot('233')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

choosen_theta=thetas[index]

j=[]
for i in data[:,0][:80]:
	y=0
	for k in range(16):
			y+=choosen_theta[k]*(i**k)
	j.append(y)

j=list(map(lambda x: x/max(j)+0.5,j))




p15 = np.poly1d(np.polyfit(data[:,0][:80], j, 15))
axes = plt.gca()
axes.set_ylim([-2,3])
plt.scatter(data[:,0][:80],data[:,1][:80])
x=np.linspace(-2,3,1000)
plt.plot(x, p15(x), '-',color="green")

plt.title("polynomial fit")


plt.show()


