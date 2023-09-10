import numpy as np
from scipy.stats import bernoulli


def boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables):
    """ A method for generating artificial data. The features are boolean, and from the uniform distribution,
        while the underlying function (concept) is a conjunction of disjunctions
        e.g.: X, Y = boolean_concept_uniform_distribution([[1,2],[1,3]], 0.5, 10, 4)
            Args:
             conjunctions:  (list of lists)-  a list of lists of features indices- each inner list represents a disjunction on the specified variables; 
             p:  (float from [0,1])-  the probability that a feature is '1'
             dataset_size: (int) - size of the dataset  
             no_variables: (int) - no. of features
            Returns:                                         
              X (2  dimensional np array)- each row represents the feature for an instance
              Y (1  dimensional np array)- the labels
    """
    X = bernoulli.rvs(p, size=(dataset_size, no_variables))
    Y = np.zeros(dataset_size)
    for conjunction in conjunctions:
        conjunction_values = np.prod(X[:, conjunction], axis=1)
        Y = Y + conjunction_values
    Y = np.clip(Y, 0.0, 1.0)
    return X, Y


