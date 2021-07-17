import numpy as np
from scipy.stats import bernoulli

def boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables):
    X = bernoulli.rvs(p, size =(dataset_size,no_variables))
    Y = np.zeros(dataset_size)
    for conjunction in conjunctions:
            conjunction_values = np.prod(X[:,conjunction], axis=1)
            Y  = Y + conjunction_values
    Y =  np.clip(Y, 0.0, 1.0)
    return X, Y

#X, Y = boolean_concept_uniform_distribution([[1,2],[1,3]], 0.5, 10, 4)
