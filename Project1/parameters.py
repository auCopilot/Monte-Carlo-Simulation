import numpy as np
### (a), (b) ###

m = 1
# delta, gamma is a column vector of size m
delta = np.ones((m, 1)) * 2
gamma = np.ones((m,m))
x = 10
sigma_s = np.ones((m, m))

### (c) ###

# Q measure paramters
mu_q = np.ones(m) * 2
sigma_q = sigma_s

# P measure parameters
mu_p = np.zeros(m)
sigma_p = sigma_s

######## (d) ########
import seaborn as sns
from scipy.stats import norm

Nd = 10000

SIMULATION_RUNS = 30
plot_n = np.arange(1, SIMULATION_RUNS + 1)

methods = ["Crude Monte Carlo", "Control Variable", "Importance Sampling"]
colors = sns.color_palette("colorblind", 3)

def confidence_interval(s, l , N, level = 0.05):
    """Compute confidence interval, equation (1.2)"""
    z_alpha = norm.ppf(1 - level)
    return (l - z_alpha*s / np.sqrt(N), l + z_alpha*s / np.sqrt(N) )
