"""Another implementation of CE.
All work belongs to: https://github.com/apourchot/CEM-RL
"""

import numpy as np

try:
    from env.tetris import Tetris
    env = Tetris(10,10)
    print('Imported Tetris')
except ImportError:
    print('Tetris nai')


class sepCEM():
    # num_params is the wieghts for the state ?
    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):
        
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)

        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite individuals
        self.elitism  = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        # when generating random variables instead of using purely random values, the alg
        # generates paris of random values with opposite signs to reduce variace in the objective function
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        # weights for each actor in population i.e. each policy
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        # inds = candidate random vectors applied with pertubation to current mean
        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds
    
    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        # negate scores as we are maximising instead of minimising - lowest scores are better
        scores *= -1
        # ranking
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        # calculate new mean by taking weighted mean of top performing parameters
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]
        # print(f'mu: {self.mu}')

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        tmp = self.weights @ (z * z)
        beta = self.num_params * self.damp / np.sum(tmp)
        tmp *= beta

        alpha = 1
        self.cov = (alpha * tmp + (1 - alpha) *
                    self.damp * np.ones(self.num_params))

        print(f'damp, beta, max_cov: {self.damp}, {beta}, {np.max(self.cov)}')

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        print(f'covariance: {self.cov}')

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


if __name__ == "__main__":
    es = sepCEM(num_params=8, mu_init=0, elitism=True, pop_size=100)
    # lines cleared plot each generation
    L_plot = []
    checkpoint = 5

    max_iterations = 40
    for i in range(max_iterations):
        # generate candidate parameters
        candidates = es.ask(pop_size=100)

        # evaluate candidate parameters
        # scores = [env.simulation(candidate_params) for candidate_params in candidates]

        scores = [np.mean([env.simulation(candidate_params) for k in range(10)]) for candidate_params in candidates]
        
        
        # update distribution parameters
        es.tell(candidates, scores)

        # mean lines cleared over 30 games, using mean of best performing
        L_mean = np.mean([env.simulation(es.mu) for k in range(30)])
        print(f'mean over 30 games: {L_mean}')
        L_plot.append(L_mean)    
        # final_mean - is the mean of the best parameter vectors
        final_mean, final_cov = es.get_distrib_params()
        print(f'iteration: {i}, final_mean: {final_mean}, final_cov: {final_cov}')
    

        if i % checkpoint == 0:
            np.save('data/sepCEM10x10/lines_cleared_{}.npy'.format(checkpoint), L_plot)
