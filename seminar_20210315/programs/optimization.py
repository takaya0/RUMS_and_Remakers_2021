import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

ITERRATION_NUMBER = 100
step_size = 0.01


def objective_function(X):
    x = X[0]
    y = X[1]
    value = 2 * x * x + 4 * x * y + 5 * y * y
    return value


def Retraction(x, tagent_vector):
    return (x + tagent_vector)/np.linalg.norm(x + tagent_vector)


def get_gradient(points):
    Gradient = gradient(points)
    gradient_on_Sphere = projection(Gradient, points)
    return gradient_on_Sphere


def gradient(points):
    Gradient = [4 * (points[0] + points[1]), 4 * points[0] + 10 * points[1]]
    Gradient = np.array(Gradient)
    return Gradient


def projection(gradient, points):
    projection = gradient - np.dot(points, gradient.T) * points
    return projection


def main():
    plt.figure(figsize=(6.0, 4.5))
    points = np.array([1, 0])
    log = []
    points_log = []

    for i in range(ITERRATION_NUMBER):
        log.append(objective_function(points))
        points_log.append(points)
        grad = gradient(points)
        next_points = Retraction(x=points, tagent_vector=step_size * grad)

        points = next_points
        if i % 10 == 0:
            print("今の座標 = ({}, {})".format(points[0], points[1]))
    points_log = np.array(points_log)
    plt.plot(log)
    plt.title("minimize $f(x, y) = 2x^2 + 4xy + 5y^2$ on unit circle")
    plt.xlabel("Iteration number")
    plt.ylabel("value of function")
    plt.show()


if __name__ == "__main__":
    main()
