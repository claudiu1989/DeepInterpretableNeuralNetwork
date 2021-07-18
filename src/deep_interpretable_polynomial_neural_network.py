import numpy as np
from scipy.optimize import minimize
from enum import Enum
from itertools import combinations_with_replacement

class GrowthPolicy(Enum):
    ALL_TERMS = 1
    GROW = 2
    PRUNE_AND_GROW = 3

class DeepInterpretablePolynomialNeuralNetwork:
    def __init__(self, d_max, lambda_param, balance, fixed_margin=True, ro=1.0, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                 max_no_terms_per_iteration=-1, max_no_terms=-1, growth_policy=GrowthPolicy.ALL_TERMS):
        # The margin
        self.ro = ro
        # Indicator of the type of margin used
        self.fixed_margin = fixed_margin
        # Positive labels (1) weight
        self.balance = balance
        # Regularisation parameter 
        self.lambda_param = lambda_param
        # Maximum degree
        self.d_max = d_max
        # Maximum degree
        self.growth_policy = growth_policy
        # The minimum magnitude of the derivative required to add the new term
        self.derivative_magnitude_th = derivative_magnitude_th
        # The minimum magnitude of a parameter (beta_i) required to keep it's assocaited term
        self.coeff_magnitude_th = coeff_magnitude_th
        
        # Maximum number of terms that can be added in an iteration
        self.max_no_terms_per_iteration = max_no_terms_per_iteration
        # Maximum number of terms
        self.max_no_terms = max_no_terms
        # Init
        self.set_to_default()

    def train(self, X_train, Y_train):
        """ The method  for model training.
            Args:
             X_train:  (2 dimensional np array with float values)- the training data input values; each row corresponds to a data point; 
                                                                   all values must be from the interval [0,1]  
             Y_train:  (1 dimensional np array with float values)- the training data labels; each entry corresponds to a data point; 
                                                                   all values must be from the set {0,1}  
            Returns:                                         
             -
        """
        # Initialisation
        self.n = len(X_train[0])
        self.m = len(Y_train)
        self.X_train = self.add_negated_variables(X_train)
        # Transform the labels from {0,1} to {-1,1}
        self.Y_train = 2.0*Y_train - 1
        if self.growth_policy == GrowthPolicy.ALL_TERMS:
            self.terms = self.generate_all_terms(self.d_max)
            self.cr_degree = self.d_max
            max_no_iterations = 1
        else:
            self.terms = self.generate_all_terms(1)
            self.cr_degree = 1
            max_no_iterations = self.d_max
        self.X_train_cr = self.compute_features(self.X_train)
        self.no_features = len(self.X_train_cr[0])
        i = 1
        self.beta_optimal = np.zeros(self.no_features)
        new_terms_added = True
        terms_of_cr_degree_were_kept = True
        while i <= max_no_iterations and terms_of_cr_degree_were_kept and new_terms_added:
            # Phase 1
            print(f'Start phase 1 of iteration {i}')
            self.beta_optimal = self.train_phase1()
            sum_beta = np.sum(self.beta_optimal)
            if not self.fixed_margin:
                if sum_beta > 0.0:
                    self.ro = 1.0/sum_beta
                    self.w_optimal =self.ro*self.beta_optimal
                else:
                    print(f'Warning: The optimal vector of weights is an all-zeros vector.')
                    self.w_optimal = self.beta_optimal
                    self.ro = 0.0
            else:
                self.w_optimal = self.ro*self.beta_optimal
            # Phase 2
            if self.growth_policy == GrowthPolicy.GROW or self.growth_policy == GrowthPolicy.PRUNE_AND_GROW:
                print(f'Start phase 2 of iteration {i}')
                if self.cr_degree < self.d_max:
                    if self.growth_policy == GrowthPolicy.PRUNE_AND_GROW:
                        terms_of_cr_degree_were_kept = self.prune_terms_and_features()
                    if len(self.terms) < self.max_no_terms or self.max_no_terms < 0:
                        new_terms_added = self.add_terms_and_features_of_next_degree(self.cr_degree + 1)
                        self.cr_degree = self.cr_degree + 1
                    else:
                        # Increase counter
                        i = i + 1
                        break
            # Increase counter
            i = i + 1
        print('\nThe training process finished with success!\n')

    def train_phase1(self):
        """ The first phase of the training iteration.
            Args:
             - 
            Returns:
            1 dimensional np array with float values- the optimal parameters values
        """
        if not self.fixed_margin:
            bnds = tuple([(0.0, None)]*self.no_features)
            res = minimize(DeepInterpretablePolynomialNeuralNetwork.objective_function, 
                           args=(self.X_train_cr, self.Y_train, self.m, self.n, self.cr_degrees_limits, self.cr_degree, self.lambda_param, self.fixed_margin, self.ro, self.balance), x0=self.beta_optimal, bounds=bnds)
        else:      
            bnds = tuple([(0.0, 1.0/float(self.ro))]*self.no_features)
            cons = ({'type': 'eq', 'fun': lambda beta: 1.0/float(self.ro) - np.sum(beta)})
            res = minimize(DeepInterpretablePolynomialNeuralNetwork.objective_function, 
                           args=(self.X_train_cr, self.Y_train, self.m, self.n, self.cr_degrees_limits, self.cr_degree, self.lambda_param, self.fixed_margin, self.ro, self.balance), x0=self.beta_optimal, bounds=bnds, constraints=cons)
        beta_optimal = res.x
        return beta_optimal

    def __predict(self, X_test):
        """ Compute predictions for new data, represented in the features space.
            Args:
             X_test: (1 dimensional np array with float values)- the test data, in the features space
            Returns:
             Two 1 dimensional np arrays with float values- the binary (0/1) predictions and the "scores" (values from [0,1])
        """
        Y_predicted = X_test.dot(self.w_optimal)
        Y_predicted_binary = np.around(Y_predicted)
        return np.array(Y_predicted_binary), Y_predicted
    
    def predict(self, X_test):
        """ Compute predictions for new data.
            Args:
             X_test: (1 dimensional np array with float values)- the test data
            Returns:
             Two 1 dimensional np arrays with float values- the binary (0/1) predictions and the "scores" (values from [0,1])
        """
        X_test_with_negated = self.add_negated_variables(X_test)
        X_test_features = self.compute_features(X_test_with_negated)
        Y_predicted_binary, Y_predicted = self.__predict(X_test_features)
        return Y_predicted_binary, Y_predicted

    def objective_function(w, X_train_cr, Y_train, m, n, cr_degrees_limits, cr_degree, lambda_param, fixed_margin, ro, balance):
        """ Computes the objective function.
            Args:
            w: (1 dimensional np array with float values) the current parameters ("beta", in the paper).
            X_train_cr: (2 dimensional np array with float values) current training data features
            Y_train: (1 dimensional np array with float values, 1.0 and -1.0)  training data labels
            m: (int) no. data points
            n: (int) no. of variables
            cr_degrees_limits: (list of integers) the start index for the terms of each degree 
            (e.g. [2,5] means that from 0 to 1 the terms have a fixed degree, and from 2 to 4 the terms have a higher degree)
            cr_degree: (int) max current degree
            lambda_param: (float) the regularization hyperparameter
            fixed_margin: (bool) if true, a fixed margin is used
            ro: (float) the margin; used only if fixed_margin=True or if the computed margin is invalid
            balance: (float) the balance hyperparameter
            Returns:
            float, the value of the objective function
        """
        balanced_Y_train = np.array([balance * y if y > 0.0 else y for y in Y_train])
        cr_ro = ro
        sum_w = np.sum(w)
        if not fixed_margin and sum_w >= 0.00001:
            cr_ro = 1.0/sum_w
        exponent = balanced_Y_train * (X_train_cr.dot(w) - 1.0/(2.0*cr_ro))
        exponent = 1 - 2.0 * exponent
        data_terms = np.exp(exponent)
        data_part = (1.0/float(m))*np.sum(data_terms)
        ant_i = 0
        wd = []
        for i in cr_degrees_limits:
            wd_i = np.sum(w[ant_i:i])
            wd.append(wd_i)
            ant_i = i
        wd = np.array(wd)
        d = np.array(range(cr_degree))+1.0
        regularization_factors = np.sqrt((d/float(m))*np.log(2.718*(2.0*n + d - 1)/d))
        regularization_part = wd.dot(regularization_factors)
        return data_part + lambda_param * regularization_part

    def compute_features_data_point(self, x):
        """ Compute all features for a data point.
            Args:
              x: (1 dimensional np array)- a data point
            Returns:
              1 dimensional np array- all features for the current data point
        """
        x_features = []
        for term in self.terms:
            feature = np.prod(x[term])
            x_features.append(feature)
        return np.array(x_features)

    def compute_features(self, X):
        """ Compute all features for the dataset
            Args:
              x: (2 dimensional np array)- a dataset
            Returns:
              2 dimensional np array- all features, for all data points
        """
        # For each data point in set X
        X_features = []
        for x in X:
            x_feat = self.compute_features_data_point(x)
            X_features.append(x_feat)
        return np.array(X_features)

    def add_new_features(self, new_terms):
        """ Add the features for the new terms.
            Args:
              new_terms: (list of lists of integers) the new terms
            Returns:
              -
        """
        # todo- implement more efficiently
        for new_term in new_terms:
            self.X_train_cr = np.array([np.append(x_train_cr, np.prod(x_train[new_term])) for x_train_cr,x_train in zip(self.X_train_cr,self.X_train)])
   
    def generate_all_terms(self, cr_d_max):
        """ Generate all terms having degree at most cr_d_max.
            Args:
              cr_d_max (int) the maximum degree
            Returns:
              list of lists of integers- the list of terms
        """
        # All variables indices, including the negates
        indexes = range(2*self.n)
        all_terms = []
        for cr_d in range(cr_d_max):
            all_terms_fixed_degree = list(combinations_with_replacement(indexes,cr_d+1))
            all_terms.extend([list(term) for term in all_terms_fixed_degree])
        # Compute the indices ranges for terms of each degree- todo: remove this part and store the indices during generation
        cr_degree = 1
        self.cr_degrees_limits = []
        for i,term in enumerate(all_terms):
            if len(term) > cr_degree:
                self.cr_degrees_limits.append(i)
                cr_degree = len(term)
        self.cr_degrees_limits.append(len(all_terms))
        return all_terms
    
    def add_terms_and_features_of_next_degree(self, next_degree):
        """ Add the terms of next degree, and the corresponding features.
            Args:
              next_degree (int) the degree of the terms that will be added
            Returns:
              bool- true if new terms were added, false otherwise
        """
        # All variables indices, including the negates
        indexes = range(2*self.n)
        # Add the terms of next degree
        new_terms = []
        # Add the non-zero derivative values 
        derivative_values = []
        # Compute the max no of terms that can be added
        maximum_no_terms_to_add = min(self.max_no_terms_per_iteration, self.max_no_terms - len(self.terms))
        if maximum_no_terms_to_add > 0:
            maximum_no_terms_to_add = min(self.max_no_terms_per_iteration, self.max_no_terms - len(self.terms))
            # Compute and cache the values needed to compute the partial derivative
            # for all new term
            data_exp_factors = self.compute_exp_factors_derivative()
            start_index_max_degree_terms = 0
            if len(self.cr_degrees_limits)>1:
                start_index_max_degree_terms = self.cr_degrees_limits[-2]
            # todo: improve efficiency
            for term in self.terms[start_index_max_degree_terms:]:
                for i in indexes:
                    new_term = term.copy()
                    # Create the new term
                    new_term.append(i) 
                    # Check if it is already in the list (the terms are sorted)
                    new_term.sort()
                    are_identical = False
                    no_new_terms = len(new_terms)
                    k = 0
                    while k < no_new_terms and not are_identical:
                        are_identical = np.array_equal(np.array(new_terms[k]), np.array(new_term))
                        k = k + 1
                    if not are_identical:
                        derivative_value = self.compute_derivative(new_term, next_degree, data_exp_factors)
                        if derivative_value < 0 and np.abs(derivative_value) > self.derivative_magnitude_th:
                            # Add the derivative value
                            derivative_values.append(derivative_value)
                            # Add the new term
                            new_terms.append(new_term)          
            
            new_terms_to_add = []
            if len(new_terms) <= maximum_no_terms_to_add:
                new_terms_to_add = new_terms
            if len(new_terms) > maximum_no_terms_to_add:
                new_terms_to_add = self.get_top_terms_by_derivative(new_terms, derivative_values, maximum_no_terms_to_add)
            self.terms.extend(new_terms_to_add)
            # Set to zero the coefficients of the new terms
            no_new_terms = len(new_terms_to_add)
            self.beta_optimal = np.append(self.beta_optimal, np.zeros(no_new_terms))
            self.w_optimal = np.append(self.w_optimal, np.zeros(no_new_terms))
            # Compute the new features and add them to self.X_train_cr
            self.add_new_features(new_terms_to_add)
            # The new number of features
            self.no_features = len(self.terms)
            # Compute the indices ranges for terms of each degree, if new terms were added
            if new_terms_to_add:
                cr_degree = 1
                self.cr_degrees_limits = []
                for i,term in enumerate(self.terms):
                    if len(term) > cr_degree:
                        self.cr_degrees_limits.append(i)
                        cr_degree = len(term)
                self.cr_degrees_limits.append(len(self.terms))
                # New terms were added, continue training
                return True 
            else:
                # No new terms, can stop now
                return False
        else:
            # No new terms
            return False

    def get_top_terms_by_derivative(self, new_terms, derivative_values, no_terms_to_return):
        """ Get the terms corresponding to the features that have the biggest (in absolute value) derivatives
            Args:
             new_terms (list of lists of integers) the new terms
             derivative_values (1 dimensional np array with negative float values) the partial derivatives
             no_terms_to_return (int) no of terms to return
            Returns:
              list of lists of integers- a subset of no_terms_to_return terms from new_terms
        """
        derivative_values_new_terms = list(zip(derivative_values, new_terms))
        # Sort by derivative value (which are negative), in ascending order
        derivative_values_new_terms.sort()
        return [derivative_term[1] for derivative_term in derivative_values_new_terms[:no_terms_to_return]]

    def prune_terms_and_features(self):
        """ Remove the features that have a small contribution
            Args:
             -
            Returns:
             -
        """
        indices_to_keep = self.beta_optimal >= self.coeff_magnitude_th
        self.beta_optimal = self.beta_optimal[indices_to_keep]
        self.w_optimal = self.w_optimal[indices_to_keep]
        terms = np.array(self.terms)[indices_to_keep]
        self.terms = [list(term) for term in terms]
        self.X_train_cr = self.X_train_cr[:,indices_to_keep]
        max_degree = 0
        for term in self.terms:
            if len(term) > max_degree:
                max_degree = len(term)
        if max_degree < self.cr_degree:
            return False
        else:
            return True
       
    def compute_derivative(self, term, next_degree, data_exp_factors):
        """ Compute the partial derivative with respect to a feature ('term')
            Args:
             term: (list of integers) the variables in the current term
             next_degree: (int) the degree of the term
             data_exp_factors ( 1 dimensional np array with float values) the values computed by 'compute_exp_factors_derivative'
            Returns:
             (float) the value of the partial derivative
        """
        new_features = [np.prod(x[term]) for x in self.X_train]
        y_ix_dki = self.Y_train * new_features
        data_part = (-2.0/float(self.m))*np.dot(y_ix_dki, data_exp_factors)
        regularization_part = np.sqrt((next_degree/float(self.m))*np.log(2.718*(2.0*self.n + next_degree - 1)/next_degree))
        return data_part + self.lambda_param * regularization_part
    
    def compute_exp_factors_derivative(self):
        """ Compute the 'exponential' factors that appear in the partial derivative. They are common for all terms, at a given iteration
            Args:
            -
            Returns:
            1 dimensional np array with float values- the values of the factors for all training points
        """
        balanced_Y_train = np.array([self.balance * y if y > 0.0 else y for y in self.Y_train])
        cr_ro = self.ro
        sum_w = np.sum(self.beta_optimal)
        if not self.fixed_margin and sum_w >= 0.00001:
            cr_ro = 1.0/sum_w
        exponent = balanced_Y_train * (self.X_train_cr.dot(self.beta_optimal) - 1.0/(2.0*cr_ro))
        exponent = 1 - 2.0 * exponent
        data_exp_factors = np.exp(exponent)
        return data_exp_factors

    def add_negated_variables(self, X):
        """ Adds the 'negated' variables.
            Args:
            X: 2 dimensional np array with float values.

            Returns:
            2 dimensional np array with float values
        """
        X_all = []
        for x in X:
            new_x = []
            neg_x = 1.0 - x 
            new_x.extend(x)
            new_x.extend(neg_x)
            X_all.append(new_x)
        return np.array(X_all)

    def get_the_model_representation(self, coefficient_significance_threshold, precision=-1):
        """ Create a string representation of the generated model.
            Args:
            coefficient_significance_threshold: (float)- a number from [0,1]; 
                                                all terms with coefficients below coefficient_significance_threshold will be ignored.
            precision: (int) - if positive, it represents the no. of decimals used for the coefficients representation; if -1, no rounding is performed

            Returns:
             string- a representation of the model
        """
        rounded_w_optimal = self.w_optimal
        if precision>0:
            rounded_w_optimal = np.around(self.w_optimal, decimals=precision)
        terms_coeffs = zip(self.terms, rounded_w_optimal)
        model_string = ''
        for term, coeff in terms_coeffs:
            term_string = ''
            if coeff >= coefficient_significance_threshold:
                for variable_index in term:
                    if variable_index < self.n:
                        if model_string == '':
                            term_string += f'({coeff})x{variable_index}'
                        elif term_string == '':
                            term_string += f'+({coeff})x{variable_index}'
                        else:
                            term_string += f'*x{variable_index}'
                    else:
                        if model_string == '':
                            term_string += f'({coeff})(1-x{variable_index-self.n})'
                        elif term_string == '':
                            term_string += f'+({coeff})(1-x{variable_index-self.n})'
                        else:
                            term_string += f'*(1-x{variable_index-self.n})'
                model_string += term_string
        if model_string == '':
            print('Warning: all coefficients are below the threshold!')
        return model_string

    def set_to_default(self):
        """ Set to default data/internal variables
            Args:
            -
            Returns:
            -
        """
         # Training data- these will remain fixed during training
        self.X_train = np.array([])
        self.Y_train = np.array([])
        # Test data - these will remain fixed during testing
        self.X_test = np.array([])
         # Current training data- data with the current features - it will remain fixed during training
        self.X_train_cr = np.array([])
        # Current test data- data with the current features - it will remain fixed during test
        self.X_test_cr = np.array([])
        # No. original features (variables)
        self.n = 0
        # No. of features (monomials)
        self.no_features = 0
        # No. of training data points
        self.m = 0
        # Optimal value of the parameters
        self.w_optimal = np.array([])
        # Optimal value of the parameters, devided by the margin
        self.beta_optimal = np.array([])
        # The current list of terms
        self.terms = []
        # The entry i (starting with 0) holds the index of the first monomial of degree i+1
        self.cr_degrees_limits = []
        # Internal variable that keeps the maximum degree of the current terms
        self.cr_degree = 1