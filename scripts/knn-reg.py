import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

np.random.seed(1337)

x = 10 * np.random.random(20)
xx = np.linspace(x.min(), x.max())
y = np.sin(x)
plt.scatter(x, y)
plt.plot(xx, np.sin(xx))

style = dict(size=10, color='gray')

xn = 7
distance = np.abs(x - xn) # menghitung jarak dengan semua instances (absolut)
neighbours = np.array([distance, y]).T
print(neighbours.shape)
for k in range(1, 4):
	print(neighbours[neighbours[:, 0].argsort()][:k]) # k-NN
	yn = neighbours[neighbours[:, 0].argsort()][:k][:,1].mean() # mean value
	plt.scatter([xn], [yn])
	plt.text(xn - .75, yn, '%d-NN' % k, **style)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('knn-reg.png')
plt.show()