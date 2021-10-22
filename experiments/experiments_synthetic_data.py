import sys
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.generate_synthetic_data import boolean_concept_uniform_distribution
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

def experiment_all_terms_degree1():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1],[3]]
    p = 0.5
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 1
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_all_terms_degree2():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2],[3, 4]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_all_terms_degree3():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2,10],[3, 4,8]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 3
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree1():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1],[3]]
    p = 0.5
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 1
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree2():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2],[3, 4]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree3():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2,10],[3, 4,8]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 3
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree4():
    # Data 
    conjunctions = [[1,2,10, 15],[1, 4,8, 12]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 4
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_and_pruning_degree5():
    # Data 
    conjunctions = [[1,3,10, 15, 16],[2, 4,8, 12]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 5
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.GROW
    max_no_terms_per_iteration = 20
    max_no_terms = 20
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=max_no_terms_per_iteration, max_no_terms=max_no_terms, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_and_pruning_degree2():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2],[3, 4]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.PRUNE_AND_GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_and_pruning_degree3():
    ''' In paper: table 2, line
    '''
    # Data 
    conjunctions = [[1,2,10],[3, 4,8]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 3
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.PRUNE_AND_GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_and_pruning_degree4():
    # Data 
    conjunctions = [[1,2,10, 15],[1, 4,8, 12]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 4
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.PRUNE_AND_GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')



def experiment_growth_and_pruning_degree5():
    # Data 
    conjunctions = [[1,3,10, 15, 16],[2, 4,8, 12]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 5
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.PRUNE_AND_GROW
    max_no_terms_per_iteration = 20
    max_no_terms = 20
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=max_no_terms_per_iteration, max_no_terms=max_no_terms, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_all_terms_degree2_no_fixed_margin():
    # Data 
    conjunctions = [[1,2],[1,3]]
    p = 0.5
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 2.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = False
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=10, max_no_terms=200, growth_policy=GrowthPolicy.ALL_TERMS)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree2_no_fixed_margin():
    # Data 
    conjunctions = [[1,2],[1,3], [2,4]]
    p = 0.5
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=10, max_no_terms=200, growth_policy=GrowthPolicy.GROW)
    
    # Evaluation
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree2_no_fixed_margin():
    # Data 
    conjunctions = [[1,2],[1,3], [2,4]]
    p = 0.5
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=10, max_no_terms=200, growth_policy=GrowthPolicy.GROW)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')


def experiment_no_growth_degree4():
    # Data 
    conjunctions = [[1,2,10, 15],[1, 4,8, 12]]
    p = 0.75
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 4
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')



def experiment_growth_and_pruning_degree10():
    # Data 
    conjunctions = [[1,2,6,7,8, 10, 15, 16, 17, 18, 19]]
    p = 0.9
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 4
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.01
    growth_policy=GrowthPolicy.PRUNE_AND_GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree10():
    # Data 
    conjunctions = [[1,2,6,7,8, 10, 15, 16, 17, 18, 19]]
    p = 0.8
    dataset_size = 1000
    no_variables = 20
    X, Y = boolean_concept_uniform_distribution(conjunctions, p, dataset_size, no_variables)

    # Model
    d_max = 10
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    coeff_magnitude_th = 0.0
    growth_policy=GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=coeff_magnitude_th, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

if __name__ == '__main__':
    experiment_all_terms_degree3()