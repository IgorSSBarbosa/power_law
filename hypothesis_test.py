from pathlib import Path
import numpy as np
import json
import argparse

from matplotlib import pyplot as plt
from scipy.optimize import linprog

class Hypothesis_test():
    def __init__(self, args):
        self.simulation = args["data_path"]
        self.data_path = 'simulation_data/'+ self.simulation
        self.basex = args["basex"]
        self.basey = args["basey"]
        self.steps = args["steps"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--data_path",
            type=str,
            default = 'srw/srw09-13-2025_13-37-40',
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
            "--steps",
            type=int,
            default = 100,
        )
        
    def compute_estimators(self,data):
        mean = np.mean(data, axis=1)
        var = np.var(data,axis=1, ddof=1)
        return mean, var
    
    ############ min max problem #################
    
    def solve_min_max(self,logn_x, log_n):
        """" 
        This function solves Min_{α,c} Max_{k} ρ^{k} |log(X_k) - α k - c|
        
        Parameters:
        -----------

        logn_x : array_like
            An array containing the values of log(X_k)

        log_n : array_like
            An array determining the range of k

        Returns:
        --------
        (t, alpha_ast, c_ast, k_ast): (np.float, int)
            t: The solution of min max,
            alpha_ast, k_ast: (One) argument of the minmax solution
        
        """

        N = len(logn_x)
        assert N == len(log_n), "logn_x and log_n must have the same length"

        # Create restrictions
        c = [0,0,1]

        for k in range(N):
            if k == 0:
                A_ub = np.array([[log_n[k], 1, -1/(self.rho**log_n[k])],
                                 [-log_n[k], -1, -1/(self.rho**log_n[k])]])
                b_ub = np.array([logn_x[k],
                                 -logn_x[k]])
            else:
                A_ub = np.vstack([A_ub, [log_n[k], 1, -1/(self.rho**log_n[k])],
                                  [-log_n[k], -1, -1/(self.rho**log_n[k])]])
                b_ub = np.hstack([b_ub, logn_x[k],
                                  -logn_x[k]])
                
        res = linprog(c, A_ub=A_ub, b_ub=b_ub)
                
        assert res.success == True, "Linear programming solved sucessefully"
        
        return res.x
        
    def re_escaled_error(self, alpha, c, k, logn_x):
        return abs(logn_x - alpha * k - c) * (self.rho ** k)


    def hypothesis_test(self, dom, X_n, alpha_0, k_0=None):
        """"
        Compute the probability of the data perform a power law, 
        that means X_n ~ n^α
        """

        # Convert inputs to numpy arrays and take log
        X_n = np.array(X_n)
        log_X_n = np.emath.logn(self.rho, X_n)
        log_n = np.emath.logn(self.rho, dom)
        I = range(len(log_n))
        I_log_n = zip(I,log_n)

        alpha_ast, log_c, t = self.solve_min_max(log_X_n, log_n)
        alpha_ast = 0.5  # fixing alpha = 0.5
        log_c = 0 # fixing c = 0
        print(f"alpha_0 = {alpha_0}, alpha_ast = {alpha_ast}, c = {log_c}, t = {t}")

        re_scale_errors = [self.re_escaled_error(alpha_ast, log_c,k, log_X_n[i]) for (i,k) in I_log_n]

        fig, ax = plt.subplots(figsize=(8,6))
        # plot the re-scaled errors
        ax.plot(dom, re_scale_errors, color= '#004A87', marker='o', label=f'$|X_k - {alpha_ast:.3f} k - {log_c:.3f}| \\rho^k$')
        ax.set_xscale('log', base=self.basex)
        ax.set_xlabel('n (log scale)')
        ax.set_ylabel('re-escaled errors')
        ax.set(title=f'Re-escaled errors of {self.simulation}')
        ax.legend()

        ax.grid()
        ax.grid(which='minor', color='0.9')

        image_path = Path('images/re-escaled_errors')/f'{self.simulation}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        plt.savefig(image_path)

    def run(self):
        metadata_path = self.data_path + '/metadata.json'
        with open(metadata_path, 'r') as f:
            json_file = json.load(f)
        results_path = self.data_path + '/log_logplot_results.json'
        with open(results_path, 'r') as file:
            results = json.load(file)
        simulation_path = self.data_path + '/simulations.npy'
        data = np.load(simulation_path)
        dom = json_file['n_variation']
        self.rho = json_file['rho']
        alpha = results['alpha_log_logplot']
        X_n, var = self.compute_estimators(data)

        self.hypothesis_test(dom, X_n, alpha_0=alpha)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Initialization Arguments')
    Hypothesis_test.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    tester = Hypothesis_test(vars(args))

    tester.run()