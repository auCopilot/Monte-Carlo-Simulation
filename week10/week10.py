import numpy as np
def robbins_monro(func, sampler, theta0,
                  a=None, max_iter=10**4,
                  tol=1e-6, dist_depend = False):

    theta_old = theta0
    convergence = False

    # Step sizes
    if a is None:
        a = [1/n for n in range(1, max_iter + 2)]

    for i in range(max_iter):

        # 1. Generate random observations
        if dist_depend:
            X = sampler(theta_old)
        else:
            X = sampler()

        # 2. Estimate the function H
        H = np.mean(func(X, theta_old))

        # 3. Update theta
        theta = theta_old - a[i] * H


        # 4. Check convergence
        if np.linalg.norm(theta - theta_old) < tol:
            convergence = True
            break

        theta_old = theta

    if convergence:
        print("Convergence at iteration", i + 1)
    else:
        print("Maximum iterations reached without convergence")

    return theta

def keifer_wolfowitz(dfunc, sampler, theta0, h, a=None, max_iter=10**4, tol=1e-6):

    theta = theta0
    convergence = False
    if a is None:
        a = [1/n for n in range(1, max_iter + 2)]

    finite_diff = lambda x, x_tilde, theta, update_vector, h, a: a * (dfunc(x, theta + h*update_vector / 2) -
                                           dfunc(x_tilde, theta - h*update_vector / 2)) / h

    for i in range(max_iter):
        X, Xtilde = sampler()
        theta_old = theta.copy()

        for j in range(len(theta)):
            e = np.zeros(len(theta))
            e[j] = 1

            theta[j] = theta_old[j] - finite_diff(X,
                                                         Xtilde,
                                                         theta_old,
                                                         e,
                                                         h[i],
                                                         a[i])


        if np.linalg.norm(theta - theta_old) < tol:
            convergence = True
            break
    if convergence:
        print("Convergence at iteration", i + 1)
    else:
        print("Maximum iterations reached without convergence")
    return theta