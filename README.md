# ExplainableAI
Materials for researches in Explainable AI

### Basic Tasks
#### HS_titanic.ipynb
- Related Paper: [https://proceedings.mlr.press/v162/agarwal22b.html](https://proceedings.mlr.press/v162/agarwal22b.html) It is a up-to-date method called Hierarchical Shrinkage which improves the accuracy and interpretability of tree-based models.
- Keys:
	- Use HSCART that they develop to fit for cleaned_titanic data (columns: 'Age', 'Pclass','Sex', 'PassengerId', 'Survived')
	- Calculate shap value, permutation importance
	- Conclusion: Feature2 indeed has greater importance value than normal CART without HS; Their HSCART is different from .tree in sklearn, so we should use `shap.KernelExplainer`
	- Need to solve: how they compute MDI?
#### TuningRF_titanic_Lai.ipynb
- Task: tune the parameters of a random forest to optimize log loss and find the best value for `min_samples_leaf`, which minimizes the log loss on a hold-out set (oob or CV). Then, compute log loss only on oob data using the best parameter.
- Method: GridSearchCV: [https://scikit-learn.org/stable/modules/grid_search.html](https://scikit-learn.org/stable/modules/grid_search.html)
- Keys:
	- def log_loss_score
	-  Generate cleaned_titanic data (columns: 'Age', 'Pclass','Sex', 'PassengerId', 'Survived')
	-  replicate the simulation design used by Strobl et al (2007) where a binary response variable Y is predicted from a set of 5 predictor variables that vary in their scale of measurement and number of categories. The first predictor variable X1 is continuous, while the other predictor variables X2,…,X5 are multinomial with 2,4,10,20 categories, respectively.
	- Power Simulation
### ConferenceSubmission
Notebooks and data for "Approximation of SHAP values for Randomized Tree Ensembles"
#### Data
All data generated and needed for import
#### Figures
All figures generated
### Notebooks
#### ti&shap_RFR.ipynb
- def generate_ti_shap(i, data) (find the best parameters each)
- Print the Waterfall Plot Individually
- Figure out which features affect the difference between TI and Shap
#### shap_ti_rmse.ipynb
- def generate_rmse_totalimportance(data, hparam, ti_rs, shap_values, it)
- Generate ti and shap and store results in /Data
- Generate filtered result
- Calculate pearsonr and spearmanr among rmse, Im_shap and Im_ti
- The correlation between explainability score and loss (here is rmse) should be negative, because the more important the features are used for prediction, the less loss it will cause. Otherwise the importance is not correctly estimated, which are fb and solar
- shap.plots.beeswarm(shap_values) for fb and solar
- Try to store the dictionary(result) into a .pickle file
#### Fig1_2.ipynb
- Multi-way AND function verifying figure 5
- Simulate random forest with multi-way AND function
- Simulate a figure similar to figure 4 with 2, 3, 4 ways and use random forest average as explanation score
	- fit model with a single tree
	- fit model with a random forest
- Multi-way OR function
	- fit model with a single tree
	- fit model with a random forest
- Sage: https://github.com/iancovert/sage
#### Other R scripts
### Simulation_SHAP
#### Simulation_SAGE.py
- Use `argparse` to change parameters remotely
- def SimulateData_simple
- fit xgboost to calculate sage and store them as .csv
- compare sage among x_train from simulation
#### Simulation.ipynb
- def oob_regression_r2_score
code:
https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
- def generate_sample_indices
- def generate_unsampled_indices
- def SimulateData_Strobl
- def SimulateData_simple
- plot power simulation for OOB and Inbag
- plot NULL simulation for OOB and Inbag
### RF_from_scratch
Reference: https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/Video%2010%20-%20Regression%201.ipynb
#### Data
#### utils
only `helper_function.py` is used for construction in `random_scratch.py`
#### Parallel Processing (could be used in ensemble)
https://www.machinelearningplus.com/python/parallel-processing-python/
```python
# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = [pool.apply(function, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()    
```