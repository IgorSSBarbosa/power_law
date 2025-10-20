import argparse
import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
from utils import compute_power_law_exponent
class LoglogPlotter():
    def __init__(self,args):
        self.simulation = args["data_path"]
        self.data_path = 'simulation_data/'+ self.simulation
        self.basex = args["basex"]
        self.basey = args["basey"]
        self.plot_cancelation = args["plot_cancelation"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--data_path",
            type=str,
            default='srw/srw08-15-2025_10-43-24'
        )
        parser.add_argument(
            "--basex",
            type=float,
            default=4,
        )
        parser.add_argument(
            "--basey",
            type=float,
            default=4,
        )
        parser.add_argument(
            "--plot_cancelation",
            action='store_true',
            help="If set, plot the Richardson cancelation heatmap",
        )

    def Richardson_step(self,dom,X_n,j):
        R = []
        J = len(X_n) -1 
        for i in range(J):
            n1, n2 = dom[i],dom[i+1]
            f1, f2 = X_n[i],X_n[i+1]
            extrapol = ((n2**j)*f2 - (n1**j)*f1)/(n2**j - n1**j)
            R.append(extrapol)

        new_dom = [dom[i+1] for i in range(J)]

        return R, new_dom
    
    def richardson_matrix(self, dom, X_n):
        J = len(dom)
        delta = np.zeros(J-1)
        R = np.zeros((J-1,J-1))
        # Calculate de inclination of each step of the log-log plot
        for i in range(J-1):
            delta[i] = (np.emath.logn(self.rho,X_n[i+1]) - np.emath.logn(self.rho,X_n[i])) / ( np.emath.logn(self.rho,dom[i+1]) - np.emath.logn(self.rho,dom[i]) )
        j=0
        while len(dom)>0:
            aux = np.zeros(J-1)
            for i in range(len(delta)):
                aux[i] = delta[i]
            R[j] = aux
            j+=1
            delta, dom = self.Richardson_step(dom, delta,j)
        return R
    
    def log_richardson_step(self, log_n, log_X_n, j):
        R = []
        J = len(log_X_n) -1 
        for i in range(J):
            k1, k2 = log_n[i],log_n[i+1]
            f1, f2 = log_X_n[i],log_X_n[i+1]
            extrapol = ((self.rho**((k2-k1)*j))*f2 - f1)/((self.rho**((k2-k1)*j)) - 1)
            R.append(extrapol)

        new_dom = [log_n[i+1] for i in range(J)]

        return R, new_dom
    
    def log_logplot_richardson(self, dom, X_n):
        '''
        This algorithm makes the 'inverse' order of Richardson matrix,
        First it uses Richardson cnacelation then it takes linear Regression
        '''
        J = len(dom)
        # Linear regression in log space
        log_n = np.emath.logn(self.rho,dom)
        log_X_n = np.emath.logn(self.rho,X_n)
        output = []
        output_r_squared = []

        j=0
        while len(log_n)>1:
            slope, intercept, r_value, _, _ = linregress(log_n, log_X_n)
            output.append(slope)
            output_r_squared.append(r_value ** 2)

            j=+1
            log_X_n, log_n = self.log_richardson_step(log_n, log_X_n, j)

        return output, output_r_squared

    def exponential_fit(self, x, a):
        x_data = np.array(x, dtype=float)
        return a * np.pow(float(self.rho), -x_data)


    def better_cancelation(self, R, a_0, plot_cancelation=False):
        '''
        This function takes the Richardson matrix and returns the best cancelation
        By analyzing the values of w[j,k] = ρ^{kj}(R[j,k] - R[j,k-1])/(1 - ρ^{J})
        '''
        J = R.shape[1]
        w = np.zeros((J-2,J-2))
        best_index = (0,0)

        for j in range(1,J-1):
            for k in range(1,J-j-1):
                w[j-1,k-1] = (R[j,J-j-1] - R[j,k-1])
        # plotting w_j as a function of k, where different j are different colors
        
        # Fit w_j vs k to a function f(k) = a*\rho^{-kJ} using mse
        plt.figure(figsize=(16, 12))
        for j in range(1,3):
            # Convert range to numpy array for the x data
            x_data = np.array(range(1, J-j-1))
            y_data = w[j-1, :J-j-2]
            popt, _ = curve_fit(self.exponential_fit, x_data, y_data)
            a = popt[0]
            plt.plot(x_data, self.exponential_fit(x_data, a), linestyle='--', label=f'Fit j={j}, a={a:.3e}')
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('w[j,k]')
            plt.title('Richardson Cancelation Weights Fit')
            plt.grid()

        if plot_cancelation:
            for j in range(1,3):
                plt.plot(range(1,J-j-1), w[j-1,:J-j-2], marker='o', label=f'j={j}')
            plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
            plt.xlabel('k')
            plt.ylabel('w[j,k]')
            plt.title('Richardson Cancelation Weights')
            plt.legend()
            plt.grid()
            plt.show()

        return R[best_index[0], best_index[1]], best_index

    def exponent_confidence_interval(self, data, alpha, confidence):
        """
        Calculate the confidence interval for the power law exponent α.
        Suppose \hat{α} ~ N(α, σ²)
        then σ²_k ~ (ν²_{ρ^k} + ν²_{ρ^{k-1}})/sqrt(numb_simul) 

        where ν²_{ρ^k} = (var(data[k,:]))/(mean(data[k,:])**2)

        Parameters:
        -----------
        dom : array_like
            Array of system sizes (e.g., simulation sizes).
        X_n : array_like
            Array of mean values for each system size.
        var : array_like
            Array of variances for each system size.
        alpha : array_like
            Estimated power law exponent for each system size.
        a_0 : float
            Intercept from the linear regression in log-log space.
        numb_simul : int
            Number of simulations used to compute the mean and variance.            
        confidence : float, optional
            Confidence level for the interval (default: 0.95).
        
        -----------
        Returns:
        ci : tuple
            Each entry is a tuple of lower and upper bounds of the confidence interval for the exponent α_k.
        """
        n, numb_simul = data.shape
        nu = np.zeros(n)
        for k in range(n):
            nu[k] = (np.sqrt(data[k,:].var(ddof=1)))/(data[k,:].mean())

        # Compute the variance of the estimated exponents
        sigma = np.zeros(n-1)
        for k in range(1, n):
            sigma[k-1] = (nu[k] + nu[k-1]) / np.sqrt(numb_simul)

        z = norm.ppf((1 + confidence) / 2)
        ci = []
        for k in range(len(sigma)):
            margin_of_error = z * sigma[k]
            ci.append((alpha[k] - margin_of_error, alpha[k] + margin_of_error))

        return ci

    def plot(self):
        '''
        Load data from the specified path, compute estimators, power law exponent,
        Richardson matrix, and log-log plot with Richardson cancelation.
        '''
        # Load metadata
        metadata_path = self.data_path + '/metadata.json'
        with open(metadata_path, 'r') as f:
            json_file = json.load(f)

        # Load simulation data
        simulation_path = self.data_path + '/simulations.npy'
        data = np.load(simulation_path)
        # Define parameters from metadata
        dom = json_file['n_variation']
        self.rho = json_file['rho']
        self.model = json_file['model']

        # The variance will be used to compute confidence intervals 
        X_n = np.mean(data, axis=1)
        var = np.var(data, axis=1, ddof=1)

        alpha, r_squared, a_0 = compute_power_law_exponent(dom, X_n, plot=True)
        print(f'alpha = {alpha}')
        print(f'r² = {r_squared}')

        R = self.richardson_matrix(dom,X_n)
        ci = self.exponent_confidence_interval(data, R[0,:], 0.95)
        
        output, output_r_squared = self.log_logplot_richardson(dom, X_n)
        print(R)
        print(output)
        
        best,best_index =self.better_cancelation(R, a_0, self.plot_cancelation)
        print(f'Best Richardson cancelation: {best}, at index {best_index}')

        results_dict = {
            'alpha_log_logplot': float(alpha),
            'Richardson_Matrix': R.tolist(),
            'Log_logplot_with_cancelation':output,
            'Log_logplot_with_cancelation_r_squared': output_r_squared,
            'Best_Richardson_cancelation': float(best),
            'Best_Richardson_cancelation_index': best_index,
        }

        # save results 
        results_path = self.data_path + "/log_logplot_results.json"
        print(f'save data in: {results_path}')
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Initialization Arguments")
    LoglogPlotter.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    plotter = LoglogPlotter(vars(args))

    plotter.plot()

    