import xgboost
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import random
import shap
import sage
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Change parameters remotely
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mtry", type=int, default=None,
                    help="max_features (mtry)")
parser.add_argument("--max_depth", type=int, default=None,
                    help="max_depth")
parser.add_argument("--type", type=str, default="train-test",
                    help="train-train or train-test")
parser.add_argument("--savename", type=str, default="simulation_sage_tt_default.csv",
                    help="save name")
parser.add_argument("--relevance", type=int, default=0.15,
                    help="relevance")


args = parser.parse_args()

mtry = args.mtry
max_depth = args.max_depth
savename = args.savename
type = args.type
relevance = args.relevance

# Similation
def SimulateData_simple(n=120, # number of rows in data
                        # M=100, # number of simulations
                        # nCores = M, # number of cores to use; set to 1 on Windows!
                        relevance=0.15, # signal srength (0 for NULL)
                        # correctBias = c(inbag=TRUE,outbag=TRUE),
                        verbose=0):

    x1 = np.random.randn(n)
    x2 = np.random.randint(1, 3, n)
    x3 = np.random.randint(1, 5, n)
    x4 = np.random.randint(1, 11, n)
    x5 = np.random.randint(1, 21, n)
    # y = np.random.binomial(n = 1, p = 0.5 + [-1,1][x2[0]-1] * relevance, size = n)
    y = np.array([])
    for i in range(n):
        y = np.append(y, np.random.binomial(n = 1, p = 0.5 + [-1,1][x2[i]-1] * relevance, size = 1))

    x_train = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

    return(x_train.to_numpy(), y)

# X_null, Y_null = SimulateData_simple(n=400, relevance=0)
result_accuracy = []
columns = ["x1", "x2", "x3", "x4", "x5"]
result_sage = pd.DataFrame(columns=columns)
start_time = time.time()
for times in range(100):
    X_train, y_train = SimulateData_simple(n=400, relevance=relevance)
    X_test, y_test = SimulateData_simple(n=400, relevance=relevance)

    # fit model no training data
    model = XGBClassifier(max_features=mtry, max_depth=max_depth)
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print(times)
    print("max_features", model.get_params()['max_features'])
    print("max_depth", model.get_params()['max_depth'])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    result_accuracy.append(accuracy)
    imputer = sage.MarginalImputer(model, X_train)
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')
    if type == "train-train":
        sage_values = estimator(X_train, y_train)
    if type == "train-test":
        sage_values = estimator(X_test, y_test)
    print(sage_values.values)
    result_sage.loc[times] = sage_values.values
    print("--- %s seconds ---" , (time.time() - start_time))
    print()

print("--- %s seconds ---" , (time.time() - start_time))

# Store best parameters in a csv
print(savename)
result_sage.to_csv(Path.cwd().joinpath(savename))
