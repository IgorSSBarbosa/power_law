import numpy as np

''' Create a fake power law to test the hypothesis test with uniformly distributed noise'''

def power_law(k: np.float128, alpha: np.float128 = 0.5, c: np.float128 = 1) -> np.float128:
    return c * (k**alpha)

def noise_power(k: np.float128, alpha: np.float128 = 0.5, c: np.float128 = 1, noise_level: np.float128 = 0.1) -> np.float128:
    y = power_law(k, alpha, c)
    noise = np.random.uniform(-noise_level, noise_level)
    return y + noise

if __name__=="__main__":
    ''' Example of usage '''
    k = np.float128(18)
    print(noise_power(k)) 