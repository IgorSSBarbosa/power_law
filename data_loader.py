# This program selects a model and saves the results of simulations
import argparse
import json
import os
from pathlib import Path
from time import localtime, strftime, time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from meta_time_algorithm import simulation_manager

# Importing models
from models.srw import srw
from models.urw import urw
from models.percolation import largest_cluster_size as perc
from models.sample import simulate as sample
from models.fake_power import noise_power, power_law

class DataLoader():
    def __init__(self, args):
        model_dict = {
            'srw': srw,
            'urw': urw,
            'percolation': perc,
            'sample': sample,
            'noise_power': noise_power,
            'power_law': power_law,
        }
        self.model_name = args["model"]
        self.model = model_dict[self.model_name]
        self.rho = args["rho"]
        self.time_budget = args["time_budget"]
        self.savedir = args["save_directory"]
        self.J = args['j']
        self.pre_simulation_budget = args['pre_simulation_budget']
        self.seed = args["seed"]
        self.batch_size = args["batch_size"]
        self.numb_simul = args["numb_simul"]
        self.domain = args["domain_interval"]
        if (self.domain is not None):
            self.domain = [int(k) for k in self.domain.split(',')]
            if (self.rho is not None):
                assert isinstance(self.domain,list) and len(self.domain)==2, "When using rho provide a list of two elements [k1,k2]"
                # Sampling in logarithm scale
                self.domain = [ np.pow(self.rho, k) for k in range(self.domain[0], self.domain[1]+1 )]  

        self.numb_threads = args["num_threads"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model", 
            type=str,
            choices = ["percolation", "srw", "urw","rwre","sample","noise_power","power_law"],
            default = "srw",
            help = "Chooses a model to simulate, the options are: \n\n "
            "'percoltion' - to bond percolation in 2 dimensions, \n\n "
            "'srw' - to Simple Random walk in 1 dimension, \n\n"
            " 'urw' - Uniform step random walk in 1 dimension, \n\n "
            "'rwre' - Random Walk in Random invironment, it is a biased random walk over a 1 dimension diffusion exclusion process \n\n"
            "'sample' - to run the sample recursive algorithm"
            "'noise_power' - to run a fake power law with noise"
            "'power_law' - to run a fake power law without noise"
        )
        parser.add_argument(
            "--rho",
            type=float,
            default=None,
            help="The base of the exponential binning, use only when the mode of bin is logarithmic",
        )
        parser.add_argument(
            "-tb",
            "--time_budget",
            type=float,
            default= 10,
            help= "Budget time, in seconds, approximally to total time to make the simulations"
        )
        parser.add_argument(
            "-save",
            "--save_directory",
            type=str,
            default="simulation_data",
        )
        parser.add_argument(
            "--numb_simul",
            "-ns",
            type=int,
            default=None,
            help="Number of simulations to run, if None it will be determined by the meta algorithm", 
        )
        parser.add_argument(
            "--domain_interval",
            '-di',
            type=str,
            default=None,
            help="list of two integers [k1,k2] determining the interval of n to simulate,"
            " if rho is provided the values will be rho^k for k in [k1,k2] or provide the full list of n values if rho is None." 
            "Default is None, which means the interval will be determined by the meta algorithm.",
        )
        parser.add_argument(
            "--j",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--pre_simulation_budget",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=100,
            help= "Random seed to reproducibility of simulations",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=100,
        )
        parser.add_argument(
            "--num_threads",
            type=int,
            default=12,
        )

    def save_data(self,data, dom, time_taken, numb_simul, batch_size):
        # Create a dictionary to save the data
        metadata = {
            "model": self.model_name,
            "rho":self.rho,
            "time_budget": self.time_budget, 
            "time_taken": time_taken,
            "seed": self.seed,
            "n_variation": dom,
            "number_of_simulations": numb_simul,
            "batch_size": batch_size,
            "number_of_threads": self.numb_threads,
        }

        # create diretory
        # Get the local time and format it
        now = localtime()
        now_str = strftime("%m-%d-%Y_%H-%M-%S", now)

        out_put_path = Path(self.savedir) / self.model_name / f"{self.model_name}{now_str}"
        self.model_dir = out_put_path
        print(out_put_path)

        out_put_path.mkdir(parents=True, exist_ok=True)

        metadata_path = out_put_path / "metadata.json"
        print(f'save data in: {metadata_path}')
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        simulation_path = out_put_path / "simulations"
        np.save(simulation_path, data)

    def generate_data(self, model, n, numb_simul):
        data = np.zeros(numb_simul)
        pbar = tqdm(range(numb_simul),leave=False)
        pbar.set_description(f'Sampling for {n} size ...')

        # Split work into chunks to reduce overhead
        batch_size = max(self.batch_size, numb_simul // 6*self.numb_threads)  # Adjust based on your system
        data = Parallel(n_jobs=self.numb_threads, batch_size=batch_size)(
            delayed(model)(n) for _ in pbar
        )
        
        return data
    
    def generate_full_data(self, numb_simul):
        
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        start_time = time()
        # Make sure domain is a list of distinct integers
        self.domain = list(set([int(np.round(n)) for n in self.domain]))
        self.domain.sort()
        I = len(self.domain)

        full_data = np.empty((I, numb_simul))
        progress_bar = tqdm(self.domain,leave=False)
        progress_bar.set_description('Simulation progress')

        for i,n in enumerate(progress_bar):
            full_data[i,:] = self.generate_data(self.model, n, numb_simul) # simulate for fixed sizes n
        
        end_time = time()
        time_taken = end_time - start_time
        return full_data, time_taken


def main():
    # define the arguments parser
    parser = argparse.ArgumentParser(description="Initialization Arguments")
    DataLoader.add_arguments(parser)
    args = parser.parse_args() # Use empty list to avoid command line parsing

    simulator = DataLoader(vars(args))

    if (simulator.numb_simul is None) and (simulator.domain is None):
        # Using the meta algorithm to determine numb_simul, k1 and k2
        assert simulator.rho is not None, "When using the meta algorithm provide rho"

        print(f'Runing simulation Manager to determine numb_simul, k1 and k2, for rho = {simulator.rho}')
        numb_simul, k2 = simulation_manager(simulator.model, simulator.time_budget, simulator.pre_simulation_budget)
        k1 = np.max([k2 - simulator.J,0])
        simulator.domain = [np.pow(simulator.rho,k) for k in range(k1,k2+1)]
    else:
        # Using the provided numb_simul, k1 and k2
        assert isinstance(simulator.numb_simul,int) and isinstance(simulator.domain,list), "Provide either of numb_simul is integer and domain is a list"
        print('Using provided numb_simul, k1 and k2')
        numb_simul = simulator.numb_simul
        k1 , k2 = simulator.domain[0], simulator.domain[-1]
    
    print(f'Running simulations for k1: {k1}, k2: {k2}, numb_simul: {numb_simul}')
    # Calculate Batch Size
    batch_size = max(simulator.batch_size, numb_simul // 100)  # Adjust based on your system

    data, time_taken = simulator.generate_full_data(numb_simul)
    simulator.save_data(data, simulator.domain, time_taken, numb_simul, batch_size) # saving as json the results


if __name__=='__main__':
    main()

    
        
