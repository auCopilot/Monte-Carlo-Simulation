import numpy as np
# Parameters
p = 0.9
alpha = 0.9
m = 10

def forest_fire():
    # create lattice, 1 indicates tree, 0 indicates no tree
    a = np.random.binomial(1, p, size=(m, m))
    # print("start")
    # print(a)
    while True:
        random_spot = (np.random.randint(0,m),np.random.randint(0,m))
        if a[random_spot] == 1:
            break
    # t0, we start fire in random spot
    def fire():
        return np.random.choice([1,2], p=[1-alpha,alpha])
    a[random_spot] = 2
    # As long as a tree and fire is in a
    t = 0
    while 1 in a and 2 in a:
        t += 1
        for i in range(m):
            for j in range(m):
                # If we are on a tree then check
                if a[(i, j)] == 2:
                    # Fire dies
                    a[(i, j)] = 3
                    # up
                    if  i > 0:
                        if a[(i - 1,j)] == 1:
                            # Fire maybe starts
                            a[(i - 1,j)] = fire()
                    # down
                    if i < m - 1:
                        if a[(i + 1,j)] == 1:
                            # Fire maybe starts
                            a[(i + 1, j)] = fire()

                    # left
                    if j > 0:
                        if a[(i,j - 1)] == 1:
                            # Fire maybe starts
                            a[(i,j - 1)] = fire()

                    # right
                    if j < m - 1:
                        if a[(i, j + 1)] == 1:
                            # Fire maybe starts
                            a[(i, j + 1)] = fire()

    values , counts = np.unique(a, return_counts=True)
    # print("end")
    # print(a)
    trees = counts[np.where(values == 1)][0] if np.any(values == 1) else 0
    fires = counts[np.where(values == 2)][0] if np.any(values == 2) else 0
    dead = counts[np.where(values == 3)][0] if np.any(values == 3) else 0
    # print("Timestep: ", t)
    #
    # print("Trees:", trees )
    #
    # print("Fires:", fires)
    #
    # print("Dead Trees:", dead)
    return 0 if trees != 0 else 1
n = 1000
i = 0
for _ in range(n):
    i += forest_fire()
print(i / n)