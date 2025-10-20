import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

def compute_power_law_exponent(self, dom , X_n, plot, confidence_interval=False):
    """
    Compute the power law exponent α and R² from a power-law relationship X_n ~ n^α.
    
    Parameters:
    -----------
    n : array_like
        Array of system sizes (e.g., simulation sizes).
    X_n : array_like
        Array of measured values (should follow X_n ~ n^α + a_0).
    plot : bool, optional
        If True, generates a log-log plot (default: True).
    confidence_interval : bool, optional
        If True, prints confidence intervals, over mean points
    
    Returns:
    --------
    alpha : float
        The power law exponent (slope of log(X_n) vs log(n)).
    r_squared : float
        The R-squared value of the linear fit.
    a_0 : float
        The intercept of the linear fit in log-log space.
    """
    # Convert inputs to numpy arrays
    X_n = np.asarray(X_n)
    
    # Linear regression in log space
    log_n = np.emath.logn(self.rho,dom)
    log_X_n = np.emath.logn(self.rho,X_n)
    slope, intercept, r_value, _, _ = linregress(log_n, log_X_n)
    alpha = slope
    r_squared = r_value ** 2

    # Plotting
    if plot:
        fig,ax = plt.subplots(figsize=(8, 6))
        # plot data and fit
        ax.plot(dom, X_n, color='#004A87', marker='o', label='Data')
        ax.plot(dom, np.pow(self.rho,intercept) * (dom ** slope), 'r--', 
                label=f'Fit: $X_L \\sim L^{{{alpha:.4f}}}$\n$R² = {r_squared:.4f}$') 
        ax.set_xscale('log', base=self.basex)
        ax.set_yscale('log', base=self.basey)

        ax.set_xlabel('$L$ (log scale)')
        ax.set_ylabel('$X_L$ (log scale)')
        ax.set(title=f'Log-Log plot of {self.simulation}')
        ax.legend()

        ax.grid()
        ax.grid(which='minor', color="0.9")
        save_plot_path = Path(f'images/log-log_plots/{self.model}/plot_{self.simulation}.png')
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(save_plot_path)
        plt.show()