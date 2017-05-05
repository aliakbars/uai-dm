from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1945)

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

k = 7
n_samples = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
X_test = X_test[np.random.choice(len(X_test), n_samples, replace=False)]
knn = clf.kneighbors(X_test)

fig, ax = plt.subplots(n_samples, k+1)
x = X_train[knn[1]].reshape(n_samples, k, img_rows, img_cols)
for i in range(n_samples):
	for j in range(k):
		ax[i][j].imshow(x[i,j,:,:], cmap='Greys')
		ax[i][j].axis('off')
	ax[i][k].imshow(X_test[i].reshape(img_rows, img_cols), cmap='Greys')
	ax[i][k].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
plt.savefig('mnist.png')
plt.show()