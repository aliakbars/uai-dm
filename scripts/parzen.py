from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1945)

X_train = 10 * np.random.random(size=40).reshape(20,2)
y_train = np.random.randint(2, size=20)
X_test = np.array([2.,4.5,7.,3.]).reshape(2,2)

fig, ax = plt.subplots(1, 2, figsize=(10,5))

for i in range(2):
	ax[i].scatter(X_train[np.where(y_train == 0),0], X_train[np.where(y_train == 0),1], marker='^')
	ax[i].scatter(X_train[np.where(y_train == 1),0], X_train[np.where(y_train == 1),1])
	ax[i].scatter(X_test[:,0], X_test[:,1], marker='x')

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)

knn = clf.kneighbors(X_test)
for i, test_point in enumerate(X_test):
	c = plt.Circle(test_point, knn[0][i,k-1], fill=False, linestyle='--')
	ax[0].add_artist(c)
	c = plt.Circle(test_point, 2, fill=False, linestyle='--')
	ax[1].add_artist(c)
ax[0].set_title('3-NN')
ax[1].set_title('Parzen Window')
# plt.savefig('parzen.png')
plt.show()