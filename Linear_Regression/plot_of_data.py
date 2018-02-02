import numpy as np
import matplotlib.pyplot as plt


data=np.load("linRegData.npy")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Plot of Data")

for i in data:
	plt.plot(i[0],i[1],'bo')

plt.show()