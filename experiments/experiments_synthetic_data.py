import sys
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.generate_synthetic_data import boolean_concept_uniform_distribution
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

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

def experiment_growth_degree4_no_fixed_margin():
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
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=GrowthPolicy.GROW)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

if __name__ == '__main__':
    experiment_growth_degree4_no_fixed_margin()