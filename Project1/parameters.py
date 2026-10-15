import numpy as np

### (a), (b) ###

# Set m = 1 for (a) and (b) and m = 2 for repetition of (a) and (b) with m = 2
m = 2
# delta, gamma is a column vector of size m
delta = np.ones((m, 1)) * 2
gamma = np.eye(m)

if m == 1:
    sigma_s = np.ones((m, m))
    x = 10
if m == 2:
    sigma_s = np.array([[1, 0.5], [0.5, 1]])
    x = 15

### (c) ###

# Q measure paramters
mu_q = np.ones(m) * np.sqrt(10) - 1
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
