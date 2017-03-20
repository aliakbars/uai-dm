import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

x = np.linspace(0, 1, 100)
y = -x * np.log2(x) - (1-x) * np.log2(1-x)
y_gini = 1 - (x ** 2 + (1-x) ** 2)

y[0] = 0
y[-1] = 0
plt.plot(x, y, label='entropy')
plt.plot(x, y_gini, label='gini', linestyle='--')
plt.xlabel('$p_{(+)}$')
plt.ylabel('$H(S)$')
plt.legend()
# plt.savefig('entropy.png')
plt.show()