"""
Cross entropy with decreasing noise
Code adapted from this repository:
https://github.com/corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method/blob/main/Tetris.py
"""

import math
import numpy as np
from scipy import stats


try:
    from env.tetris import Tetris
    print('Imported Tetris')
    env = Tetris(6,5)
except ImportError:
    print('Tetris not loaded')



def cross_entropy_deacreasing_noise(alpha, N_iteration,rho,a,b):
    """Run CE with decreasing noise.
    
    @param alpha:        discount rate
    @param N_iterations: number of iterations.
    @param rho:          the fraction of vectors selected
    @param a:            noise numerator calculation
    @param b:            noise denominator calculation
    """   
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
            sample = distribution.rvs() 
            temp = []
            # now for evaluating we are doing 10 games instead of 1
            for e in range(10):
                temp.append(env.simulation(sample))
            sample_score.append(np.mean(temp))
            sample_list.append(sample)
    
        # Keeping the rho*N bests vectors
        k=math.floor(N*rho)

        indices=sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        sample_high = [sample_list[i] for i in indices]
        best_sample=sample_list[indices[0]]
        best_weights.append(best_sample)

        # New parameter estimation using MLE
        mean = np.mean(sample_high, axis = 0)
        cov =  np.cov(sample_high, rowvar = False)
        cov += np.identity(cov.shape[0]) * 1e-6
        res = (mean, cov)

        L_norm.append(np.linalg.norm(cov))
        #add noise 
        noise = max(0, a-N/b)
        matrix_noise = np.diag([noise]*8)

        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])+matrix_noise))    


        L_mean=[sample_score[indices[0]]]
        for k in range (29):
            L_mean.append(env.simulation(best_sample))

        # print(np.mean(L_mean))
        L_plot.append(L_mean)
        t+=1
        # print(L_plot,L_norm,mean)
        print(f'iteration: {j}')
    return(L_plot, L_norm,mean, best_weights)

if __name__ == "__main__":
    alpha = 0.9          # Discount rate
    N_iterations = 100   # Number of iterations
    rho = 0.1            # Fraction of vectors selected
    a = 5             # exploratino in first 20 episode
    b = 100

    L_plot, _, _, best_weights = cross_entropy_deacreasing_noise(alpha, N_iterations, rho, a, b)