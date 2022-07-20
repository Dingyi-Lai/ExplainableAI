# Import standard Python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from pathlib import Path
import pyreadr
import time
import pickle
import random
import warnings
import numbers
from warnings import warn
from scipy.sparse import issparse
# from itertools import compress
# import operator
from sklearn.metrics import mean_squared_error # squared=False -> root version
path = '/Users/aubrey/Documents/GitHub/ExplainableAI/ConferenceSubmission/Data/'
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
np.random.seed(888) # keep consistent
random.seed(888)
from sklearn.utils.random import sample_without_replacement
# from ..utils import rand_uniform
# from .utils.random import sample_without_replacement

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def our_rand_r(seed: np.uint32):
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed == None or seed == 0): seed = 1

    seed *= np.uint32(seed << 13)
    seed *= np.uint32(seed >> 17)
    seed *= np.uint32(seed << 5)

    return seed % (np.uint32(0x7FFFFFFF) + 1)

def rand_uniform(low:float,high:float,random_state:np.uint32):
    """Generate a random double in [low; high)."""
    return float(((high - low) * float(our_rand_r(random_state)) /
            float(np.uint32(0x7FFFFFFF))) + low)


def check_purity(y, typ='regression'):
    
    'checks if a leaf node is perfectly pure, in other words, if the leaf node contains only one class'
    # also for regression case: min_samples_leaf = 1
    if typ == 'classification':
        unique_classes = np.unique(y)  # Count number of classes in section of data
    if typ == 'regression':
        unique_classes = y
    if len(unique_classes) == 1:  # Check if the node is pure
        return True
    else:
        return False

def classify_data(y):
    
    'classifies data according to the majority class of each leaf'
    # Only for classification case
    
    unique_classes, counts_unique_classes = np.unique(y, return_counts=True)
    # Returns classes and no. of obs per class

    index = counts_unique_classes.argmax() # Index of class with most obs
    classification = unique_classes[index] # Class chosen for classification which is class with most obs
    
    return classification

def split_data(X, y, split_column, split_value):
    
    'splits data based on specific value, will yield both a split for the features X and target y'
    
    split_column_values = X[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        X_below = X[split_column_values <= split_value]  # Partitions data according to split values from previous functions
        X_above = X[split_column_values >  split_value]
        
        y_below = y[split_column_values <= split_value]
        y_above = y[split_column_values >  split_value]
    
    # feature is categorical
    else:
        X_below = X[split_column_values == split_value]
        X_above = X[split_column_values != split_value]

        y_below = y[split_column_values == split_value]
        y_above = y[split_column_values != split_value]
    return X_below, X_above, y_below, y_above


def calculate_impurity(y, k=0, typ="regression"):
    # method="gini",
    'calculates impurity for each partition of data, either entropy or gini'
    n = len(y)
    # classification
    if (typ == "classification") or (len(np.unique(y)) == 2):
        # if method == "entropy":
        #     impurity = sum(probabilities * -np.log2(probabilities))  # Could replace with misclassification
        # if method == "gini":
        _, counts = np.unique(y, return_counts=True)

        probabilities = counts / counts.sum()  # Probability of each class
        
        impurity = 1 - sum(probabilities**2)
    if typ == "regression":
        if len(y) == 0:   # empty data
            impurity = 0
        else:
            impurity = np.mean((y-np.mean(y))**2)
        # /n??? - MSE is a bit easier to interpret
        ###### Sum of square error????
        ###### https://www.stat.cmu.edu/~cshalizi/350/2008/lectures/24/lecture-24.pdf

        #classification
        #regression : y is binary -> equal to gini
    
    # for binary case, finite sample correction, impurity is weighted by n/(n-1)
    if n>k:
        impurity = impurity*n/(n-k) # add tree_depth to argument
        # shap value (consistency problem)
    else:
        print("n<=k, error!")
    return impurity

def calculate_overall_impurity(y_below, y_above, k=0, typ="regression"):

    'calculates the total entropy after each split'

    n = len(y_below) + len(y_above)
    p_data_below = len(y_below) / n
    p_data_above = len(y_above) / n

    overall_impurity = p_data_below * calculate_impurity(y_below,typ=typ,k=k)+\
    p_data_above * calculate_impurity(y_above, typ=typ,k=k)

    return overall_impurity, n

def determine_best_split(X, y, potential_splits,typ="regression",k=0):
    
    'selects which split lowered gini/mse the most'
    first_iteration = True
    # n_final=len(y)
    overall_impurity = calculate_impurity(y,typ=typ,k=k)  # the function will loop over and replace this with lower impurity values
    overall_impurity_for_gain = overall_impurity.copy()
    best_split_column=[]
    best_split_value=[]
    # impurity = []
    n_final = []
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=value)
            # check that both children have samples sizes at least k+1! 
            # if (len(y_below) >= (k+1)) and (len(y_above) >= (k+1)): 
            current_overall_impurity, n = calculate_overall_impurity(y_below, y_above,k=k, typ=typ)
            # impurity.append(current_overall_impurity)
            # # Goes through each potential split and only updates if it lowers entropy
            # 
            if first_iteration or current_overall_impurity < overall_impurity: 
                first_iteration = False
                overall_impurity = current_overall_impurity # Updates only if lower entropy split found, in the end this is greedy search
                # best_split_column = column_index
                # best_split_value = value
                best_split_column = [column_index]
                best_split_value = [value]
                n_final = [n]
            if current_overall_impurity == overall_impurity: 
                best_split_column.append(column_index)
                best_split_value.append(value)
                n_final.append(n)

    # randomly select multiple potential splits
    record = pd.DataFrame([best_split_column,best_split_value,n_final]).transpose()
    record.columns = ['best_split_column','best_split_value','n_final']
    # breakpoint()
    result = record.sample()
    # try:
    #     result = record.sample()
    # except:
    #     # print(len(y_below), len(y))
    #     print(potential_splits)
    #     print(record)
    #     raise

    gain = overall_impurity_for_gain - current_overall_impurity
    rescale_gain = gain*result.n_final/len(y) #might only use rescale_gain
    return int(result.best_split_column), float(result.best_split_value), rescale_gain

def get_potential_splits(X, y, random_subspace = None, random_state=None, k=0, min_samples_leaf=1):
    
    'first, takes every unique value of every feature in the feature space, then finds the midpoint between each value'
    'modified to add random_subspace for random forest'
    # Get valid random state
    # random_state = check_random_state(random_state)
    potential_splits = {}
    _, n_columns = X.shape  # No need for rows, we choose the column to split on
    # Only need second value of .shape which is columns
    
    column_indices = list(range(n_columns))
    if random_subspace and random_subspace <= len(column_indices):  # Randomly chosen features
        # column_indices = random.sample(population=column_indices, k=random_subspace)
        # random_instance = check_random_state(random_state)
        column_indices = np.array(column_indices)[sample_without_replacement(n_population=len(column_indices),\
         n_samples=random_subspace, random_state=random_state)]

    for column_index in column_indices:
        potential_splits[column_index] = [] 
        values = X[:, column_index] 
        unique_values = np.unique(values)  # Get all unique values in each column

        for index in range(len(unique_values)):  # All unique feature values
            if index != 0:  # Skip first value, we need the difference between next values
                # Stop early if remaining features are constant
                # current_value = unique_values[index]
                # previous_value = unique_values[index - 1]  # Find a value and the next smallest value
                potential_split = (unique_values[index] + unique_values[index - 1]) / 2  # Find difference between the two as a potential split
                # print(random_state)
                # potential_split = rand_uniform(previous_value,current_value,random_state)
                # if potential_split == current_value:
                #     potential_split = previous_value

                # try to split the data
                _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split)
                # Reject if min_samples_leaf is not guaranteed
                # check that both children have samples sizes at least k+1! 
                if (len(y_below) >= (k+1)) and (len(y_above) >= (k+1)) and (len(y_below) >= min_samples_leaf) and (len(y_above) >= min_samples_leaf): 
                    potential_splits[column_index].append(potential_split)
                # Reject if min_samples_leaf is not guaranteed

        if potential_splits[column_index] == []:
            potential_splits = {key:val for key, val in potential_splits.items() if key != column_index}
                    # potential_splits.remove(column_index)
    # print(column_indices)
    return potential_splits

def create_leaf(y, typ):
    if typ == "classification":
        classification = classify_data(y)
    if typ == "regression":
        classification = np.mean(y)
    feature_name = [-1]
    return classification, feature_name

def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

def decision_tree_algorithm(X, y, counter=0, min_samples_leaf=1, max_depth=5, min_samples_split=2,
                            random_subspace = None, tree_num = 0,typ="regression",k=0, random_state=None):

    'same function as in the Decision Tree notebook but now we add random_subspace argument'
    # random_state = check_random_state(random_state)
    # Data preparation

    if counter == 0:  # Counter tells us how deep the tree is, this is before the tree is initiated
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = X.columns
        FEATURE_TYPES = determine_type_of_feature(X)
        X = X.values  # Change all to NumPy array for faster calculations
        y = y.values
    # If we have started the tree, X should already be a NumPy array from the code above
    potential_splits = get_potential_splits(X, y, random_subspace, random_state, k, min_samples_leaf)  # Check for all possible splits ONLY using the random subspace and not all features!    
    # Base cases
    if (check_purity(y)) or (len(y) < 2*min_samples_leaf) or (counter == max_depth) or (len(y)<min_samples_split) or potential_splits=={}:
        classification, feature_name = create_leaf(y, typ)
        return classification, feature_name

    # Recursive part
    else:
        counter += 1  # Tells us how deep the tree is
        # print(potential_splits)
        best_split_column, best_split_value, gain = determine_best_split(X, y, potential_splits,typ=typ,k=k)  # Select best split based on impurity
        # print(best_split_column, best_split_value, gain)
        X_below, X_above, y_below, y_above = split_data(X, y, best_split_column, best_split_value)  # Execute best split
        
        # # check for empty data or too few samples
        # if (min(len(y_below),len(y_above)) < min_samples_leaf):
        #     classification, feature_name = create_leaf(y, typ)
        #     return classification, feature_name
        
        # Code to explain decisions made by tree to users
        feature_name = COLUMN_HEADERS[best_split_column]
        type_of_feature = FEATURE_TYPES[best_split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, best_split_value) # Initiate explanation of split
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, best_split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        feature_gain = [[tree_num, feature_name, gain]]
        # Pull answers from tree
        yes_answer, yes_feature_gain = decision_tree_algorithm(X_below, y_below, counter, min_samples_leaf,
                                                                max_depth, min_samples_split, random_subspace,
                                                                tree_num,typ,k, random_state)
        no_answer, no_feature_gain = decision_tree_algorithm(X_above, y_above, counter, min_samples_leaf,
                                                            max_depth, min_samples_split, random_subspace,
                                                            tree_num,typ,k, random_state)

        # Ensure explanation actually shows useful information
        if yes_answer == no_answer: # If decisions are the same, only display one
            sub_tree = yes_answer
            feature_gain = yes_feature_gain
        else:
            sub_tree[question].append(yes_answer)
            feature_gain.extend(yes_feature_gain)
            sub_tree[question].append(no_answer)
            feature_gain.extend(no_feature_gain)

        return sub_tree, feature_gain

def predict_example(example, tree, counter=0):

    'takes one observation and predicts its class'
    
    if counter == 0 and isinstance(tree, dict) == False: # very shallow tree settings may only vote one way, this first if-statement takes its vote into account
        return tree
    
    else:
        counter += 1
    
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # Ask question
    if comparison_operator == "<=":
        if example[str(feature_name)] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # Feature is categorical
    else:
        if str(example[str(feature_name)]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # Base case
    if not isinstance(answer, dict):
        return answer
    
    # Recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

    
# Gathers all test data
def decision_tree_predictions(test_df, tree):
    'applies predict_example to all of the test set'
    if len(test_df) != 0:
        predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    else:
        predictions = pd.Series()
    return predictions

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices

def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(
        random_state, n_samples, n_samples_bootstrap
    )
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices

def random_forest_algorithm_oob(X, y, n_trees, n_features=None, dt_max_depth=2,typ="regression",k=0, random_state=888, oob_score =True, min_samples_split=2, min_samples_leaf=1):
    'puts the bootstrap sample in the decision tree algorithm with max depth and the random subset of features set, in otherwords, builds the forest tree by tree'
    forest = []
    feature_gain = []
    # if issparse(X):
    #     X = X.tocsr()

    n_samples = y.shape[0]


    random_instance = check_random_state(random_state)
    # We draw from the random state to get the random state we
    # would have got if we hadn't used a warm_start.
    seed = random_instance.randint(MAX_INT, size=n_trees)


    # inbag_pred = np.zeros(n_samples, dtype=np.float64)
    oob_pred = np.zeros(n_samples, dtype=np.float64)

    n_oob_pred = np.zeros(n_samples, dtype=np.int64)
    # n_inbag_pred = np.zeros(n_samples, dtype=np.int64)


    for i in range(n_trees): #loops for the amount of trees set to be in the forest   
        # for each tree, the same data y and X is used
        
        # training and testing dataset from bootstraping
        
        # about 2/3 are training or inbag
        if oob_score:
            sample_indices = _generate_sample_indices(seed[i], n_samples, n_samples)
            unsampled_indices = _generate_unsampled_indices(seed[i], n_samples, n_samples)
            
            X_inbag = X.iloc[sample_indices] # will affect the building of trees
            y_inbag = y.iloc[sample_indices]
            
            X_oob = X.iloc[unsampled_indices] # about a third of the data are test data
            # y_test = y.iloc[unsampled_indices]
            # random_seed = seed[i]
        else:
            X_inbag = X
            y_inbag = y
            X_oob = X
            # random_seed = random_instance
        # random.seed(seed[i])
        tree, feature_gain0 = decision_tree_algorithm(X_inbag, y_inbag,
                                                      max_depth=dt_max_depth,
                                                      random_subspace=n_features,
                                                      tree_num=i,typ=typ,k=k,
                                                      min_samples_split=min_samples_split, 
                                                      min_samples_leaf=min_samples_leaf,
                                                      random_state=None) #creates individual trees

        # if we only consider oob
        y_predict_oob = decision_tree_predictions(X_oob, tree)
        if oob_score:
            oob_pred[unsampled_indices] += y_predict_oob
            n_oob_pred[unsampled_indices] += 1
        else:
            oob_pred += y_predict_oob
            n_oob_pred += 1
        # # if we also consider inbag
        # y_predict_inbag = decision_tree_predictions(X_inbag, tree)
        # inbag_pred[unsampled_indices] += y_predict_inbag
        # n_inbag_pred[unsampled_indices] += 1
        # # possible weighting method
        # # values_inbag, counts_inbag = np.unique(y_predict_train, return_counts=True)
        # # y_predict_inbag = np.zeros(len(y_predict_train))
        # # n_inbag = np.zeros(len(y_predict_train))
        # # for i, g in enumerate(y_predict_train):
        # #     y_predict_inbag[i] = g*counts_inbag[values_inbag==g]
        # #     n_inbag[i] = counts_inbag[values_inbag==g]

        forest.append(tree)
        feature_gain.extend(feature_gain0)

    if oob_score:
        if (n_oob_pred == 0).any():
            warn(
                "Some inputs do not have OOB scores. This probably means "
                "too few trees were used to compute any reliable OOB "
                "estimates.",
                UserWarning,
            )
            n_oob_pred[n_oob_pred == 0] = 1
    oob_pred /= n_oob_pred
    mse_oob_pred = mean_squared_error(y,oob_pred)
    return forest, feature_gain, mse_oob_pred

        
def random_forest_predictions(test_df, forest, typ='regression'):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)  # Key for dictionary
        predictions = decision_tree_predictions(test_df, tree=forest[i])  # Predictions from trees
        df_predictions[column_name] = predictions  # Insert predictions into dictionary

    df_predictions = pd.DataFrame(df_predictions)  # Change dictionary to pandas DF
    if typ=='classification':
        random_forest_predictions = df_predictions.mode(axis=1)[0]  # Take mode of predictions over trees for final prediction
    else:
        random_forest_predictions = np.mean(df_predictions, axis=1)
    # If there is an even number of predictions, just default to the first value (very unlikely with many trees)
    
    return random_forest_predictions


# The following could be wrapped in a function
def generate_mse_fi(X,y, n_trees=1, random_state=888,oob_score = True, min_samples_split=2, min_samples_leaf=1,\
                    #n_bootstrap=data.shape[0], bootstrap_ratio=1, train_ratio=0.7, bootstrap=True\
                    n_features=None, dt_max_depth=2,typ="regression", k=0):
    # if len(y)<n_bootstrap, take the ratio
    # how to decide bootstrap_ratio???
    
    # Record the time
    start = time.time()

    # , mse_inbag, mse_oob = random_forest_algorithm(X, y,
    #                                                 n_trees=n_trees, n_features=n_features,
    #                                                 dt_max_depth=dt_max_depth,typ=typ,k=k)
    forest, feature_gain, mse_oob_pred = random_forest_algorithm_oob(X, y,n_trees=n_trees, n_features=n_features,
                                                    dt_max_depth=dt_max_depth,typ=typ,k=k, random_state=random_state,
                                                     oob_score=oob_score, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
#     # store the feature_importance
#     feature_gain_result = pd.DataFrame(columns=["tree_num","feature", "value"])
#     for i,j in enumerate(feature_gain):
#         if j != -1:
#             feature_gain_result.loc[i] = j
#             if pd.api.types.is_list_like(j[2]) ==True:
#                 breakpoint()
#                 feature_gain_result.loc[i,"value"] = j[2][0]
    
    # predict y
    y_predict_from_scratch = random_forest_predictions(X, forest)
    # y_predict_sklearn = rf.predict(X)

    # mean_squared_error
    mse_rf_prediction = mean_squared_error(y,y_predict_from_scratch)
    # mse_sklearn = mean_squared_error(y_test,y_predict_sklearn)

#     # sum the feature importance of included features
#     fi_simulation = feature_gain_result.groupby(['tree_num','feature']).sum()
#     fi_simulation_s = fi_simulation.groupby(['feature']).sum()
#     fi_simulation_s = fi_simulation_s.sort_values("value")

    print('------',time.time()-start,'s ------')
    # fi_simulation_s,
    return mse_oob_pred, mse_rf_prediction
# , mse_sklearn, fi_sklearn



######## don't need to repeat sklearn when k is not the same


def generate_mse_sklearn(X,y, random_state=888, n_estimators = 1, oob_score = True, dt_max_depth=2,n_features=None, min_samples_split=2, min_samples_leaf=1):
    # if len(y)<n_bootstrap, take the ratio
    # how to decide bootstrap_ratio???
    
    # Record the time
    start = time.time()
    
    rf = RandomForestRegressor(random_state=random_state, max_depth = dt_max_depth, max_features = n_features, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, n_estimators = n_estimators, bootstrap=oob_score, oob_score = oob_score)
    rf.fit(X, y)
    mse_rf_prediction_sklearn = mean_squared_error(y, rf.predict(X))
    if oob_score:
        mse_sklearn = mean_squared_error(y, rf.oob_prediction_)
    else:
        mse_sklearn = mse_rf_prediction_sklearn
    print('------',time.time()-start,'s ------')
    return mse_sklearn, mse_rf_prediction_sklearn

##### wrap the following 
def easy_for_test(name='cpu', n_trees=200, random_state=888, n_features=2, oob_score = True, dt_max_depth=2):

    # Read original Data
    data = pyreadr.read_r(path+'SRData.RData')
    # path2 = path + 'mse&fi/'
    # Data processing
    X0 = data[name]
    X1 = X0.select_dtypes(include=np.number).iloc[:,:-1] # numerical features exclude the last column (y)
    if len(X0.select_dtypes(include='category').columns) !=0: # recognize categorical feature
        X2 = pd.get_dummies(X0[(X0.select_dtypes(include='category')).columns], drop_first=True) # change it into one_hot_encoding
    else: X2 = pd.DataFrame()

    X = pd.concat(objs=[X2, X1], axis=1) # combine dummies and numerical features

    y = data[name].iloc[:,-1] # the last column is y
    
    random.seed(0)

    # Iterate to store ti and shap
    # mse_k0_from_scratch_inbag = []
    # mse_k0_from_scratch_oob = []

    # fi_k0_simulation_s = {}
    mse_k0_oob_pred = []
    mse_k0_pred = []
    # mse_k1_from_scratch_inbag = []
    # mse_k1_from_scratch_oob = []

    # fi_k1_simulation_s = {}
    mse_k1_oob_pred = []
    mse_k1_pred = []

    mse_k0_sklearn_oob = []
    mse_k0_sklearn = []
    print(name)
    # fi_k0_simulation_s['df_{}'.format(name)],
    mse_oob_pred, mse_rf_prediction = generate_mse_fi(X,y, k=0, n_trees=n_trees,\
        random_state=random_state, n_features=n_features, oob_score = oob_score, dt_max_depth=dt_max_depth)
    # mse_k0_from_scratch_inbag.append(mse_inbag)
    # mse_k0_from_scratch_oob.append(mse_oob)
    mse_k0_oob_pred.append(mse_oob_pred)
    mse_k0_pred.append(mse_rf_prediction)
    # fi_k1_simulation_s['df_{}'.format(name)],
    mse_oob_pred, mse_rf_prediction = generate_mse_fi(X,y, k=1, n_trees=n_trees,\
        random_state=random_state, n_features=n_features, oob_score = oob_score, dt_max_depth=dt_max_depth)
    # mse_k1_from_scratch_inbag.append(mse_inbag)
    # mse_k1_from_scratch_oob.append(mse_oob)
    mse_k1_oob_pred.append(mse_oob_pred)
    mse_k1_pred.append(mse_rf_prediction)
    # sklearn
    mse_oob, mse_rf_prediction_sklearn = generate_mse_sklearn(X,y, n_estimators=n_trees, random_state=random_state,\
         n_features=n_features, oob_score = oob_score, dt_max_depth=dt_max_depth)
    mse_k0_sklearn_oob.append(mse_oob)
    mse_k0_sklearn.append(mse_rf_prediction_sklearn)
    return mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_k0_sklearn,mse_k0_pred,mse_k1_pred
    

