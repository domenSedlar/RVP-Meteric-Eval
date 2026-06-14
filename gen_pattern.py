import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def _gauss_helper(x, A, mu):
    r = (x - mu).T @ A @ (x - mu)

def gauss(n, m, mu, sigma_x, sigma_y, rho):
    """
    Returns an (n, m) grid containing the 2D Gaussian distribution.
    Coordinates are evaluated across a pixel grid of 0 to n and 0 to m.
    """
    x_range = np.arange(n)
    y_range = np.arange(m)
    X, Y = np.meshgrid(x_range, y_range, indexing='ij')
    
    grid = np.stack([X, Y], axis=-1) # grid is (n, m, 2) (axis=-1, adds new axis to the end)
    sigma = np.array([sigma_x, sigma_y])
    xs = (grid - mu)/sigma
    
    A = np.array([
        [1 / (sigma_x**2), -rho / (sigma_x * sigma_y)],
        [-rho / (sigma_x * sigma_y), 1 / (sigma_y**2)]
    ]) / (1 - rho**2)
    
    # Compute the exponent: -0.5 * (xs^T @ A @ xs) for each pixel
    # We use einsum to multiply the (n,m,2) vector by the (2,2) matrix, 
    # and then dot it back with the (n,m,2) vector.
    A_xs = np.einsum('ijk, lk -> ijl', xs, A)     # Multiply matrix A by vectors
    exponent = -0.5 * np.einsum('ijk, ijk -> ij', xs, A_xs) # Vector-vector dot product
    
    norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    space = norm * np.exp(exponent)
    
    space /= space.sum()
    
    return space

def small():
    n, m = (6,6)
    mu1 = np.array([1,1.5])
    sigma1 = np.array([1,1.3])
    rho1 = 0

    mu2 = np.array([4,1])
    sigma2 = np.array([1.1,0.9])
    rho2 = 0.5

    A = gauss(n, m, mu1, sigma1[0], sigma1[1], rho1) + gauss(n,m, mu2, sigma2[0], sigma2[1], rho2)
    noise = np.random.normal(np.mean(A), np.average(A)*0.15, size=(n,m))
    A = A + noise
    print(A.shape)
    plt.imshow(A)
    plt.show()
    return A

def large():
    n, m = (60,60)
    mu1 = np.array([20, 25])
    sigma1 = np.array([1,1.5]) * 2.5
    rho1 = 0.5

    mu2 = np.array([40,35])
    sigma2 = np.array([1,1]) * 2.5
    rho2 = 0

    A = gauss(n, m, mu1, sigma1[0], sigma1[1], rho1)*0.7 + gauss(n,m, mu2, sigma2[0], sigma2[1], rho2)*0.3
    noise = np.random.normal(np.mean(A), np.average(A)*0.1, size=(n,m))
    A = A + noise
    print(A.shape)
    plt.imshow(A)
    plt.show()
    return A

if __name__ == "__main__":
    L = large()
    S = small()
    os.makedirs('./Matrices/Sorted/', exist_ok=True)
    np.savetxt('./Matrices/Sorted/small_gauss.tsv', S, delimiter='\t')
    np.savetxt('./Matrices/Sorted/large_gauss.tsv', L, delimiter='\t')