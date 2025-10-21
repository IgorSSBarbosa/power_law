import numpy as np

import matplotlib.pyplot as plt

def w_J(j, rho, lambda_param):
    """Calculate w_J coefficient"""
    term1 = (1/j)
    term2 = (lambda_param * (1 - lambda_param))**j
    term3 = (1 - rho**j)
    term4 = rho**(j*(j-1)/2)
    return term1 * term2 * term3 * term4

def phi_J(L, j, rho, lambda_param):
    """Calculate phi_J(L) function"""
    return w_J(j, rho, lambda_param) * L**(-j)

# Parameters
rho = 2  # example value for rho > 1
lambda_param = 2  # example value for lambda
L = np.linspace(0, 100, 10000)  # L values from 10^0 to 10^4
j_values = range(1, 10)  # plot for j = 1,2,3,4,5

# Create plot
plt.figure(figsize=(20, 12))
for j in j_values:
    y = phi_J(L, j, rho, lambda_param)
    plt.plot(L, abs(y), label=f'j={j}')

plt.grid(True)
plt.xlabel('L')
# restrict y-axis to better visualize
plt.ylim(0, 0.5)
plt.ylabel('|φ_j(L)|')
plt.title('Power Law Functions')
plt.legend()
plt.show()