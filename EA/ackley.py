import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uqtestfuns as uqtf
from typing import Sequence

def ackley(xx: list[float], a = 20, b = 0.2, c = 2 * np.pi):
    size = len(xx)
    sum1 = 0
    sum2 = 0
    for x in xx:
        sum1 += x**2
        sum2 += np.cos(c * x)
        
    term1 = -a * np.exp(-b * np.sqrt(sum1 / size))
    term2 = -np.exp(sum2 / size)
    
    return term1 + term2 + a + np.exp(1)


def ackley_batch(xx: Sequence[float], yy: Sequence[float], a = 20, b = 0.2, c = 2 * np.pi):
    


def plot_ackley(r_min = -50, r_max = 50):
    xaxis = np.arange(r_min, r_max, 2.0)
    yaxis = np.arange(r_min, r_max, 2.0)
    x, y = np.meshgrid(xaxis, yaxis)
    results = ackley([x, y])


def main():
    # Create a meshgrid of x and y values
    x = np.linspace(-32.768, 32.768, 100)
    y = np.linspace(-32.768, 32.768, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the Ackley function at each point in the meshgrid
    fun = uqtf.Ackley(3)
    Z = fun()

    # Create a 3D plot of the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Ackley(X, Y)')
    plt.show()
    
if __name__ == "__main__":
    main()
