from scipy.misc import comb
import math

def ensemble_error(n_classifer, error):
    k_start = math.ceil(n_classifer / 2.0)
    probs = [comb(n_classifer, k) * error**k * (1-error)**(n_classifer-k)
                for k in range(k_start, n_classifer+1)]
    return sum(probs)

if __name__ == '__main__':
    import numpy as np
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifer=11, error=error)
                    for error in error_range]
    import matplotlib.pyplot as plt
    plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
    plt.plot(error_range, error_range, linestyle='--', label='Base Error', linewidth=2)
    plt.xlabel('Base Error')
    plt.ylabel('Base/Ensemble Error')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()