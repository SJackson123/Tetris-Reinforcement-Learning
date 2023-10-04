"""
Cross entropy method with constant noise.
Code adapted from this repository:
https://github.com/corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method/blob/main/Tetris.py
"""

import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


try:
    from env.tetris import Tetris
    print("imported Tetris")
    env = Tetris(6,5)
except ImportError:
    print('Tetris not loaded')


def cross_entropy_constant_noise(alpha, N_iteration,rho,noise):
    """Run CE with constant noise added.
    
    @param alpha:        discount rate
    @param N_iterations: number of iterations.
    @param rho:          the fraction of vectors selected
    @param noise:        value of constant noise to add""" 
                               
    # Initialisation
    mu0 = [0]*8
    sigma0 = np.diag([100]*8)
    V0 = (mu0, sigma0)
    parameters = [V0]
    t=1

    L_plot=[]
    L_norm=[]
    best_weights = []

    for j in range (N_iteration):
        # Create the distribution
        distribution = stats.multivariate_normal(parameters[t-1][0], parameters[t-1][1])
        
        # Evaluate each parameter pool
        N = 100
        sample_list = []
        sample_score= []

        for i in range(N):
            # vector of weights
            sample = distribution.rvs() 
            sample_score.append(env.simulation(sample))
            sample_list.append(sample)
            
        # Keeping the rho*N best vectors
        k=math.floor(N*rho)

        indices=sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        sample_high = [sample_list[i] for i in indices]
        best_sample=sample_list[indices[0]]
        best_weights.append(best_sample)

        # New parameter estimates
        mean = np.mean(sample_high, axis = 0)
        cov =  np.cov(sample_high, rowvar = False)
        res = (mean, cov)

        #add noise 
        matrix_noise = np.diag([noise]*8)
        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])+matrix_noise))    

        # calculate mean of best 30 games - use 30 becuase high variance
        L_mean=[sample_score[indices[0]]] 
        for k in range (29):
            L_mean.append(env.simulation(best_sample))

        L_plot.append(L_mean)
        t+=1
        # print(L_plot,L_norm,mean)
        # print(f'L_plot: {L_plot}')
        print(f'iteration: {j}')

    return(L_plot, mean, best_weights)


if __name__ == "__main__":
    alpha = 0.9          # Discount rate
    N_iterations = 100   # Number of iterations
    rho = 0.1            # Fraction of vectors selected
    noise = 0.5

    L_plot, _ , best_weights = cross_entropy_constant_noise(alpha, N_iterations, rho, noise)