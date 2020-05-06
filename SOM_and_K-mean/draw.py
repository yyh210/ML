import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 50)
y = x ** 2
f = 2 * x
plt.plot(x, y, linestyle=':', color='y', label='y=x^2')
plt.plot(x, f, color='r', label='y=2*x')
plt.xlabel('old wang')
plt.ylabel('old yan')
plt.legend(loc='best')
plt.show()