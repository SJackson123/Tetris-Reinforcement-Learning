"""
Standard cross entropy method.
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
    env = Tetris(6, 5)
except ImportError:
    print('Tetris not loaded')


def cross_entropy(alpha, N_iteration,rho):
    """Run CE.
    
    @param alpha:        discount rate
    @param N_iterations: number of iterations.
    @param rho:          the fraction of vectors selected""" 
    
    # Initialisation
    mu0 = [0]*8
    sigma0 = np.diag([100]*8)
    V0 = (mu0, sigma0)
    parameters = [V0]
    t=1

    L_plot=[]
    L_norm=[]
    # store the best weights from each iteration
    best_weights = []

    for j in range (N_iteration):
        # Create the distribution for the weight vector
        distribution = stats.multivariate_normal(parameters[t-1][0], parameters[t-1][1],allow_singular=True)
        
        # Evaluate each parameter pool
        N = 100
        sample_list = []
        sample_score = []

        for i in range(N):
            # vector of parameters W
            sample = distribution.rvs()
            sample_score.append(env.simulation(sample))
            sample_list.append(sample)

        # Keeping the rho*N best vectors
        k=math.floor(N*rho)

        # index of best 10 samples i.e. index 0 is best in this sample
        indices=sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        # list of best random Weight vectors
        sample_high = [sample_list[i] for i in indices]
        # get best sample weights 
        best_sample=sample_list[indices[0]]
        best_weights.append(best_sample)

        # New parameter estimation using MLE
        mean = np.mean(sample_high, axis = 0)
        cov =  np.cov(sample_high, rowvar = False,bias=True)
        res = (mean, cov)

        # Frobenius norm - quantify the spread of data
        # convergence criteria
        L_norm.append(np.linalg.norm(cov))

        # update parameters
        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])))    

        # play 30 games with best sample weights
        # calulate mean of the best 30 games
        L_mean=[sample_score[indices[0]]] 
        for k in range(29):
            L_mean.append(env.simulation(best_sample))
        
        print(np.mean(L_mean))
        L_plot.append(L_mean)
        t+=1
        print(f'iteration: {j}')
        
    return(L_plot,L_norm,mean, best_weights)


if __name__ == "__main__":
    alpha = 0.9          # Discount rate
    N_iterations = 100   # Number of iterations
    rho = 0.1            # Fraction of vectors selected

    L_plot, _, _, best_weights = cross_entropy(alpha, N_iterations, rho)
    
  
