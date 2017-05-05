from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-white')

np.random.seed(1337)

x1 = 3.5 + np.random.randn(10)
x2 = 6.5 + np.random.randn(10)
y1 = 5 + np.random.randn(10)
y2 = 5 + np.random.randn(10)
x = np.append(x1, x2)
y = np.append(y1, y2)

vor = Voronoi(np.array([x, y]).T)
voronoi_plot_2d(vor, show_points=False, show_vertices=False)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter([6], [5], marker='X')
# plt.savefig('knn.png')
# plt.savefig('voronoi.png')
plt.show()