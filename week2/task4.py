import numpy as np
import matplotlib.pyplot as plt


def monte_carlo(func, n, a = 0, b = 1):
    x = np.random.uniform(a, b, n)
    return (b-a) * np.mean(func(x))

def integrator(func, n, a = 0, b = 1, method ="all"):

    delta = (b - a) / n
    # Creates intervals with (b-a) / n spacing
    intervals = np.linspace(a, b, n + 1)
    s_left = np.sum(func(intervals[:-1]) * delta)
    s_right = np.sum(func(intervals[1:]) * delta)
    s_trapezoid = np.sum((func(intervals[:-1]) + func(intervals[1:])) * delta * 0.5)



    if method == "left":
        return s_left
    elif method == "right":
        return s_right
    elif method == "trapezoid":
        return s_trapezoid
    elif method == "monte_carlo":
        return monte_carlo(func, n, a, b)
    elif method == "all":
        d = {"left": s_left, "right": s_right, "trapezoid": s_trapezoid}
        return d
    else:
        raise AttributeError("Invalid method")

# Approximate the integral int[0,1] sqrt(1-u^2) dx using the left, right, and trapezoid methods
def exact_solution(a = 0,b = 1):
    ex = lambda u: 0.5 * u * np.sqrt(1-u**2) + 0.5 * np.arcsin(u)
    return ex(b) - ex(a)
f = lambda u: np.sqrt(1-u**2)
ns =  [2**i for i in range(20)]

methods = ["left", "right", "trapezoid", "monte_carlo"]
errors = {m : ([(abs(integrator(f, n, method= m) - exact_solution()), n) for n in ns]) for m in methods}

def plot_errors(errors):
    for m in methods:
        y, x = zip(*errors[m])
        plt.loglog(x, y, label=m)
        plt.legend(loc="lower right")
        plt.xlabel("log(n)")
        plt.ylabel("log(Error)")
    plt.grid(True)
    plt.show()

plot_errors(errors)









