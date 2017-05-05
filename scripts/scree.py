from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.style.use('seaborn')

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

v = []
for i in range(1,11):
	kmeans = KMeans(n_clusters=i)
	kmeans.fit(X)
	v.append(kmeans.inertia_)

plt.plot(range(1,11), v, 'o-')
plt.xticks(range(1,11), range(1,11))
plt.xlabel('# clusters')
plt.ylabel('aggregate distances')
plt.savefig('scree.png')
plt.show()