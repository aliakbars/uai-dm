from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)

X_train = 10 * np.random.random(size=40).reshape(20,2)
y_train = np.random.randint(2, size=20)

med_x1 = np.median(X_train[:,0])
med_x20 = np.median(X_train[X_train[:,0] < med_x1,1])
med_x21 = np.median(X_train[X_train[:,0] >= med_x1,1])

fig, ax = plt.subplots(figsize=(6,6))

plt.scatter(X_train[np.where(y_train == 0),0], X_train[np.where(y_train == 0),1], marker='^')
plt.scatter(X_train[np.where(y_train == 1),0], X_train[np.where(y_train == 1),1])
plt.plot([med_x1, med_x1], [0, 10])
plt.plot([0, med_x1], [med_x20, med_x20])
plt.plot([med_x1, 10], [med_x21, med_x21])

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)

knn = clf.kneighbors(X_train[2:3,:])
c = plt.Circle(X_train[2], knn[0][0,k-1], fill=False, linestyle='--')
ax.add_artist(c)
plt.savefig('kdtree.png', bbox_inches='tight')
# plt.show()