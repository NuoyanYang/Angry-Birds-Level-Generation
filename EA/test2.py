import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from uqtestfuns import Ackley

# Create a meshgrid of x and y values
x = np.linspace(-32.768, 32.768, 100)
y = np.linspace(-32.768, 32.768, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Evaluate the Ackley function at each point in the meshgrid
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = Ackley(np.array([X[i, j], Y[i, j]]))

# Create a 3D plot of the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Ackley(X, Y)')
plt.show()