# Guideline
## VSCode for Python
### What is VSCode
Visual Studio Code, also commonly referred to as VS Code, is a source-code editor made by Microsoft with the Electron Framework, for Windows, Linux and macOS. Features include support for debugging, syntax highlighting, intelligent code completion, snippets, code refactoring, and embedded Git.
### Important Functionalities
#### Explorer: Manage Local Files
#### Search: Search Codes by Keywords
#### Source Control: Sync via Git
From local to GitHub: staged changes -> commit with messages -> push
From GitHub to local: pull -> compare the changes
#### Run and Debugs
I use it usually for Jupyter notebook, otherwise PDB is enough. Converting .ipynb to .py for debugging is recommended.
#### Remote Explorer
If remote servers with strong CPU and GPU are available, it is better to connect with remote server to run the codes if it is secure enough. 
##### Remote Server Setup
- `ssh` for connection
- `nvidia-smi` to check server status
- `virtualenv` or `conda` to set up virtual environment (`miniconda` is recommended)
- `source` to activate virtual environment
- `pip install -r requirements.txt` to install the requirements file of your project (it is recommended to conclude all necessary modules for your project)
- `python filename.py` to run the code 
##### Sync Between Local and Remote
- From Local to Remote: `rsync -urvz localfilelocation remotefilelocation`
- From Remote to Local: `rsync -vz remotefilelocation localfilelocation` or `scp -r remotefilelocation localfilelocation``
##### Key Pairs Generation for Convenience
- setup private/public key on remote server: [https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-2)
- Edit your ./ssh/config to include something like this:  
[https://linuxize.com/post/using-the-ssh-config-file/](https://linuxize.com/post/using-the-ssh-config-file/)
##### Kill .py process if I forgot to close it
`pkill -9 ython`
## Shared Repository via Github
### Git (could be accessed via VSCode)
```Terminal
git clone https://github.com/your_name/repository_name.git
```

### Jupyter Notebook
If you don't want to reload vscode every time when you changed helper_function.py for example, add the following right after importing packages
```Python
%load_ext autoreload
%autoreload 2
```

My Repository for storage: https://github.com/Dingyi-Lai/ExplainableAI
Clone it or copy it and then edit for free

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
## Log via Obsidian

Obsidian based on local server and connect it with a git private repository. Write a tiny program to sync every 30 mins(should find the tutorial in Google)
