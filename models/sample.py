# sample.py

import random
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def simulate(j):
     if (j == 0):
         return random.random()
     a = simulate(j - 1)
     b = simulate(j - 1)
     c = simulate(j - 1)
     d = simulate(j - 1)
     return min(max(a, b), max(c, d))

if __name__ == "__main__":
    import sys
    
    if (len(sys.argv) == 1):
        print("Argument n missing")
        exit(1)

    n = int(sys.argv[1])
    numb_simulations = 100000

    I = range(n)

    random.seed(datetime.now().timestamp())
    results = []
    error_bars = []

    # Split work into chunks to reduce overhead
    numb_threads = 10

    pbar = tqdm(range(numb_simulations), leave=False)

    for k in tqdm(I, leave=False): 
        batch_size = max(10, numb_simulations // 6*numb_threads)  # Adjust based on your system
        data = Parallel(n_jobs=numb_threads, batch_size=batch_size)(
            delayed(simulate)(k) for _ in pbar
        )
        results.append(np.mean(data))
        error_bars.append(np.std(data) / np.sqrt(numb_simulations))  # Standard error

    p_c = 0.5*(np.sqrt(5) - 1) # True limit

    plt.plot(I, results, marker='o')
    plt.fill_between(I, np.array(results) - np.array(error_bars), np.array(results) + np.array(error_bars), color='b', alpha=0.2)
    plt.axhline(y=p_c, color='r', linestyle='--', label='p_c: {:.4f}'.format(p_c))
    plt.xlabel('n')
    plt.ylabel('simulate(n)')
    plt.title('Simulation Results')
    plt.grid()
    
    print("Done")

    print("Results:", results)
    plt.show()

    error = [np.abs(results[i] - p_c) for i in I]
    zeta = [error[i-1]/error[i] for i in range(1, len(error))]

    # Plotting the errors
    plt.figure()
    plt.plot(I[1:], zeta, marker='o')
    plt.xlabel('n')
    plt.ylabel('Zeta')
    plt.title('Zeta Analysis')
    plt.grid()
    plt.show()
    print("Errors:", error)
    print("Zeta:", zeta)
    print("zeta estimated:", np.mean(zeta))

    plt.figure()
    plt.plot(I, error, marker='o')
    plt.xlabel('n')
    plt.ylabel('Error')
    plt.title('Error Analysis')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(I, error_bars, marker='o')
    plt.xlabel('n')
    plt.ylabel('Error Bars')
    plt.yscale('log')
    plt.title('log(Error Bars) Analysis')
    plt.grid()
    plt.show()

