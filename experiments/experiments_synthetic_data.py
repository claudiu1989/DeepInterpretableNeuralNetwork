import sys
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.generate_synthetic_data import boolean_concept_uniform_distribution
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy

def basic_experiment_no_growth():
    d_max = 2
    balance = 2.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = False
    sip = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=10, max_no_terms=200, growth_policy=GrowthPolicy.ALL_TERMS)
    X, Y = boolean_concept_uniform_distribution([[1,2],[1,3]], 0.5, 1000, 4)
    Y = 2*Y - 1
    no_runs = 1
    sip.evaluate_multiple_times(X, Y, no_runs)
    print('Optimal w:')
    w_optim_trimed = [w if w>0.01 else 0.0 for w in sip.w_optimal]
    print(w_optim_trimed)
    print(f'Ro:{sip.ro}')

if __name__ == '__main__':
    basic_experiment_no_growth()