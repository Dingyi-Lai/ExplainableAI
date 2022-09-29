from utils.helper_function import easy_for_test, decision_tree_algorithm, decision_tree_predictions,num_leaf_sklearn
import pyreadr
import random
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error
from imodels import HSTreeClassifierCV, HSTreeRegressor, HSTreeRegressorCV

###### test just decision_tree_algorithm:

# Create a decision tree
# Read original Data
data = pyreadr.read_r(Path.cwd().joinpath('ConferenceSubmission/Data/SRData.RData'))
random.seed(0)

X0 = data['boston'][0:50]
X1 = X0.select_dtypes(include=np.number).iloc[:,:-1] # numerical features exclude the last column (y)
if len(X0.select_dtypes(include='category').columns) !=0: # recognize categorical feature
    X2 = pd.get_dummies(X0[(X0.select_dtypes(include='category')).columns], drop_first=True) # change it into one_hot_encoding
else: X2 = pd.DataFrame()

X = pd.concat(objs=[X2, X1], axis=1) # combine dummies and numerical features
y = data['boston'].iloc[0:50,-1] # the last column is y

# # from sklearn import datasets
# # X,y = datasets.load_boston(return_X_y=True)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
#regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
#regr_2.fit(X, y)
# Predict
mse_sklearn = mean_squared_error(y, regr_1.predict(X))
print(mse_sklearn)

# print(num_leaf_sklearn(regr_1))
# plot_tree(regr_1)
# plt.show()
regr_2 = HSTreeRegressorCV(estimator_=regr_1)  # Applying HS to tree ensembles
regr_2.fit(X, y)   # fit model
mse_sklearn2 = mean_squared_error(y, regr_2.predict(X))
print(mse_sklearn2)
# print(regr_2)
print(regr_2.reg_param) #lambda

regr_3 = HSTreeRegressor(estimator_=regr_1, reg_param=10)  # Applying HS to tree ensembles
regr_3.fit(X, y)   # fit model
mse_sklearn3 = mean_squared_error(y, regr_3.predict(X))
print(mse_sklearn3)

# max_depth=7 mse_sklearn=3.04446734098988 num_leaf_sklearn=74
# max_depth=8 mse_sklearn=2.08803819218943 num_leaf_sklearn=122

X = pd.DataFrame(X)
y = pd.DataFrame(y)
sub_tree, feature_gain, y_above_list = decision_tree_algorithm(X, y, max_depth=2,\
    n=len(y),k=0, if_HS = True, Lambda=regr_2.reg_param)# 
y_2 = decision_tree_predictions(X, sub_tree)
# print(sub_tree)

mse_from_scratch = mean_squared_error(y, y_2)
print(mse_from_scratch)

sub_tree3, feature_gain3, y_above_list3 = decision_tree_algorithm(X, y, max_depth=2,\
    n=len(y),k=0, if_HS = True, Lambda=10)# 
y_3 = decision_tree_predictions(X, sub_tree3)
print(sub_tree)

mse_from_scratch3 = mean_squared_error(y, y_3)
print(mse_from_scratch3)


# HS comparison
# 2.08803819218943
# 3.8269002623540276
# 10.0
# 3.8269002623540276

# max_depth=7 mse_from_scratch=3.0223585198593192 num_leaf=74
# max_depth=8 mse_from_scratch=2.1467208239169704 -> 2.0880381921894298 num_leaf=113 ->129 (after reuse original split function)

# k=None: 6.0665044378656034 max_depth=8 num_leaf=49
# %%
# # a single regression tree with no feature subsampling and without bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=1, random_state=888, n_features=None, oob_score = False, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)

# ------ 5.187978982925415 s ------
# ------ 4.756967306137085 s ------
# ------ 3.89385986328125 s ------
# ------ 0.01292729377746582 s ------
# [2.08803819218943] [2.0880381921894298] [3.1042965268112543] [6.0665044378656034]
# [2.08803819218943] [2.0880381921894298] [3.1042965268112543] [6.0665044378656034]

# boston
# ------ 1.1720707416534424 s ------
# ------ 1.177666187286377 s ------
# ------ 0.0080718994140625 s ------
# [25.699467452126065] [25.69946745212606] [25.69946745212606] [25.699467452126065] [25.69946745212606] [25.69946745212606]

# bike
# ------ 0.815035343170166 s ------
# ------ 0.809049129486084 s ------
# ------ 0.009579181671142578 s ------
# [1189358.0939978634] [1189358.0939978634] [1189358.0939978634] [1189358.0939978634] [1189358.0939978634] [1189358.0939978634]

# %%
# a regression RF with no feature subsampling and without bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=50, random_state=888, n_features=None, oob_score = False, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)
# ------ 260.3922040462494 s ------
# ------ 240.88047051429749 s ------
# ------ 198.3216359615326 s ------
# ------ 0.2356281280517578 s ------    
# [2.0880381921894298] [2.0880381921894298] [3.1226655733637374] [6.0665044378656034] 
# [2.0880381921894298] [2.08803819218943] [3.122665573363736] [6.0665044378656034]

# boston
# ------ 47.22374510765076 s ------
# ------ 46.860538721084595 s ------
# ------ 0.09009408950805664 s ------
# [25.69946745212606] [25.699467452126065] [25.699467452126065] [25.69946745212606] [25.699467452126065] [25.699467452126065]

# bike
# ------ 42.48018288612366 s ------
# ------ 40.517165184020996 s ------
# ------ 0.0775609016418457 s ------
# [1189358.0939978631] [1189358.0939978631] [1189358.0939978631] [1189358.0939978631] [1189358.0939978631] [1189358.0939978631]
# %%
# a single regression tree with no feature subsampling and with bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=1, random_state=888, n_features=None, oob_score = True, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)

# [372.85851480270117] [372.98735981953405] [372.0496407457443] [372.58645821545423] 
# [8.277857465770051] [8.406702482602903] [8.154533521208874] [10.511609341408018]

# boston
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.6194779872894287 s ------
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.6391780376434326 s ------
# /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/ensemble/_forest.py:560: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.009733200073242188 s ------
# [375.30341596510385] [375.30341596510385] [375.30341596510385] [26.842521632061803] [26.84252163206181] [26.84252163206181]
# heavily overfitting

# bike
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.6110939979553223 s ------
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.5977871417999268 s ------
# /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/ensemble/_forest.py:560: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.008217811584472656 s ------
# [15911933.010625577] [15911933.010625577] [15911933.010625577] [1219151.2396910982] [1219151.2396910982] [1219151.2396910982]

# %%
# a regression RF with no feature subsampling and with bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=50, random_state=888, n_features=None, oob_score = True, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)

# ------ 144.5541639328003 s ------
# ------ 134.41313481330872 s ------
# ------ 108.18750286102295 s ------
# ------ 0.15381979942321777 s ------
# [11.855114977007311] [11.468748124739776] [12.386535727235733] [13.390450402894459] 
# [2.5680717055905986] [2.518996162200485] [3.154825318244759] [5.198276111826488]

# boston
# ------ 33.64089298248291 s ------
# ------ 32.013039112091064 s ------
# ------ 0.1029512882232666 s ------
# [23.113725992659568] [23.113725992659568] [23.135379726705654] [19.637763148775104] [19.6377631487751] [19.671261945174]

# bike
# ------ 29.468831062316895 s ------
# ------ 30.317517042160034 s ------
# ------ 0.10532474517822266 s ------
# [1099154.2934679927] [1122228.606087166] [1122228.606087166] [1022383.9089936205] [1056218.9729029455] [1056218.9729029455]
# %%
# a single regression tree with feature subsampling and no bootstrap
# After predicting also via random_forest_prediction
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=1, random_state=888, n_features=2, oob_score = False, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)
# ------ 0.8094820976257324 s ------
# ------ 0.6097729206085205 s ------
# ------ 0.47420406341552734 s ------
# ------ 0.005959987640380859 s ------
# [5.777647767265414] [6.849476349930525] [12.690171182257904] [14.048672714402647]
# [5.777647767265414] [6.849476349930525] [12.690171182257904] [14.048672714402647]

# boston
# ------ 0.19294118881225586 s ------
# ------ 0.2405540943145752 s ------
# ------ 0.004965782165527344 s ------
# [64.00768654985019] [55.56397378229071] [35.20338813330231] [64.00768654985019] [55.56397378229071] [35.20338813330231]
# bike
# ------ 0.08408498764038086 s ------
# ------ 0.15855622291564941 s ------
# ------ 0.00487518310546875 s ------
# [3285406.666936513] [1633756.9231196765] [2999702.9903363] [3285406.666936513] [1633756.9231196765] [2999702.9903363]
# %%
# a regression random forest with feature subsampling and no bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=50, random_state=888, n_features=2, oob_score = False,  dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)
# ------ 33.906386852264404 s ------
# ------ 32.604050159454346 s ------
# ------ 25.314924001693726 s ------
# ------ 0.06840777397155762 s ------
# [1.9391377274315615] [2.4713128830678066] [3.2006000105842105] [8.45041249280608] 
# [1.9391377274315615] [2.4713128830678066] [3.2006000105842105] [8.45041249280608]

# boston
# ------ 8.47632098197937 s ------
# ------ 8.238931894302368 s ------
# ------ 0.05460405349731445 s ------
# [34.99152622195074] [33.94148217126307] [33.736871763516525] [34.99152622195074] [33.94148217126307] [33.736871763516525]
# bike
# ------ 9.292362928390503 s ------
# ------ 8.274893999099731 s ------
# ------ 0.055844783782958984 s ------
# [1622534.1080286095] [1515235.801127118] [1585328.1712459957] [1622534.1080286095] [1515235.801127118] [1585328.1712459957]
# %%
# a single regression tree with feature subsampling and bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=1, random_state=888, n_features=2, oob_score = True, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)

# [372.80550207975824] [379.36879092339075] [374.9984709796863] [375.9849894859138] 
# [10.829106777077886] [18.58611507160771] [16.056781961760375] [18.11173117501624]

# boston
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.19441795349121094 s ------
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.1975100040435791 s ------
# /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/ensemble/_forest.py:560: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.0069811344146728516 s ------
# [390.9139101879343] [378.48363172552337] [382.88047865339416] [65.58817754717896] [36.21329278356018] [44.193573380353]

# bike
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.14415621757507324 s ------
# /Users/aubrey/Documents/GitHub/ExplainableAI/RF_from_scratch/utils/helper_function.py:539: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.22140908241271973 s ------
# /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/ensemble/_forest.py:560: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
#   warn(
# ------ 0.007792234420776367 s ------
# [16637684.897450801] [16252209.341190526] [16288705.71473319] [3320010.928588815] [2223570.1010203105] [2246869.7104294654]
# %%
# a regression random forest with feature subsampling and bootstrap
# mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred = \
#     easy_for_test('boston', n_trees=50, random_state=888, n_features=2, oob_score = True, dt_max_depth=8)
# print(mse_k0_sklearn_oob,mse_k0_oob_pred,mse_k1_oob_pred,mse_kd_oob_pred,\
#         mse_k0_sklearn,mse_k0_pred,mse_k1_pred,mse_kd_pred)

# ------ 25.31430673599243 s ------
# ------ 23.02707004547119 s ------
# ------ 18.30252504348755 s ------
# ------ 0.09483504295349121 s ------
# [13.48705211655512] [13.288608621637158] [12.82528733977663] [17.664684059340725] 
# [3.831165999568835] [4.358166210970921] [5.018965290664204] [9.93110151768885]

# boston
# ------ 5.320831060409546 s ------
# ------ 6.1016151905059814 s ------
# ------ 0.08919787406921387 s ------
# [39.486221920196286] [40.10359672742968] [33.07004259840589] [35.1108060075375] [35.34053562619086] [30.392251319069164]

# bike
# ------ 7.567225933074951 s ------
# ------ 6.042450904846191 s ------
# ------ 0.09042716026306152 s ------
# [1672896.77493479] [1563830.322253841] [1541431.9924319147] [1592436.8681786775] [1483593.223293898] [1464048.148547606]
# %%

# Read original Data
# data = pyreadr.read_r(Path.cwd().joinpath('ConferenceSubmission/Data/SRData.RData'))
# path2 = path + 'mse&fi/'

# dataname = ['abalone', 'bike', 'boston', 'concrete', 'cpu', 'csm', 'fb', 'parkinsons','servo', 'solar','synthetic1','synthetic2'] # real data
# # dataname = ['bike'] # real data
# # dataname = ['cpu'] # real data
# ######## use some of them

# # Iterate to store ti and shap
# # k0_sklearn_oob_bike = []
# # k0_oob_pred_bike = []
# # k1_oob_pred_bike = []
# mse_k0_sklearn_oob_final = []
# mse_k0_oob_pred_final = []
# mse_k1_oob_pred_final = []
# mse_kd_oob_pred_final = []
# mse_k0_sklearn_final = []
# mse_k0_pred_final = []
# mse_k1_pred_final = []
# mse_kd_pred_final = []
# for index, name in enumerate(dataname):
    
#     temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8 = easy_for_test(name=name,n_trees=50, random_state=888,\
#         n_features=2, oob_score = True, dt_max_depth=8)
#     mse_k0_sklearn_oob_final.append(temp1)
#     mse_k0_oob_pred_final.append(temp2)
#     mse_k1_oob_pred_final.append(temp3)
#     mse_kd_oob_pred_final.append(temp4)
#     mse_k0_sklearn_final.append(temp5)
#     mse_k0_pred_final.append(temp6)
#     mse_k1_pred_final.append(temp7)
#     mse_kd_pred_final.append(temp8)

# # %%
# k_group = (['sklearn','k=0','k=1','k=TreeDepth'])*2
# oob_all = (['oob'])*4+(['all'])*4
# mse_summary = pd.DataFrame([mse_k0_sklearn_oob_final,mse_k0_oob_pred_final,mse_k1_oob_pred_final,mse_kd_oob_pred_final,\
#     mse_k0_sklearn_final,mse_k0_pred_final,mse_k1_pred_final,mse_kd_pred_final])
# mse_summary.columns = dataname
# mse_tags = pd.DataFrame([k_group,oob_all]).T
# mse_tags.columns = {'k_group','oob_all'}
# mse_summary = pd.concat([mse_summary,mse_tags], axis=1)
# print(mse_summary)

# # # # %% [markdown]
# # # # k=1 increases mse: concrete, (cpu), csm, fb, parkinsons, synthetic1 (multiplicative term)

# # # %%
# # # store the results, don't need to run it again
# # # mse_summary.to_csv(path_save+'mse_summary1.csv')
# mse_summary.to_csv(Path.cwd().joinpath("RF_from_scratch/Data/mse_summary_k.csv"))
# mse_summary2: thredshold for categorical features is 15
# mse_summary3: thredshold for categorical features is 2
# Store tree in a pickle
# with open(Path.cwd().joinpath('mse_k0_from_scratch_train.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(mse_k0_from_scratch_train, f, pickle.HIGHEST_PROTOCOL)

# with open(Path.cwd().joinpath('mse_k1_from_scratch_train.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(mse_k1_from_scratch_train, f, pickle.HIGHEST_PROTOCOL)
    
# with open(Path.cwd().joinpath('mse_k0_from_scratch_test.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(mse_k0_from_scratch_test, f, pickle.HIGHEST_PROTOCOL)

# with open(Path.cwd().joinpath('mse_k1_from_scratch_test.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(mse_k1_from_scratch_test, f, pickle.HIGHEST_PROTOCOL)

# with open(Path.cwd().joinpath('fi_k0_simulation_s.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(fi_k0_simulation_s, f, pickle.HIGHEST_PROTOCOL)

# with open(Path.cwd().joinpath('fi_k1_simulation_s.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(fi_k1_simulation_s, f, pickle.HIGHEST_PROTOCOL)


# %%
# recall the data
# mse_summary = pd.read_csv(Path.cwd().joinpath("mse_summary.csv")).iloc[:,1:]

# with open(Path.cwd().joinpath('mse_k0_from_scratch_train.pickle'), 'rb') as f:
#     mse_k0_from_scratch_train = pickle.load(f)
    
# with open(Path.cwd().joinpath('mse_k1_from_scratch_train.pickle'), 'rb') as f:
#     mse_k1_from_scratch_train = pickle.load(f)

# with open(Path.cwd().joinpath('mse_k0_from_scratch_test.pickle'), 'rb') as f:
#     mse_k0_from_scratch_test = pickle.load(f)
    
# with open(Path.cwd().joinpath('mse_k1_from_scratch_test.pickle'), 'rb') as f:
#     mse_k1_from_scratch_test = pickle.load(f)
    
# with open(Path.cwd().joinpath('fi_k0_simulation_s.pickle'), 'rb') as f:
#     fi_k0_simulation_s = pickle.load(f)
    
# with open(Path.cwd().joinpath('fi_k1_simulation_s.pickle'), 'rb') as f:
#     fi_k1_simulation_s = pickle.load(f)

# %%
# check the result
# mse_summary

# %%
# fi_k0_simulation_s



# # %%
# k0_sklearn_oob

# # %%
# k0_oob_pred

# # %%
# k1_oob_pred

# # %%
# # # Grow RF_ experiments
# # # forest, feature_gain = random_forest_algorithm(X, y, n_trees=50, n_bootstrap=800, n_features=8, dt_max_depth=3)

# # forest, feature_gain = random_forest_algorithm(X_train, y_train,bootstrap=False, n_trees=50, n_bootstrap=400, n_features=2,
# #                                                dt_max_depth=2,method="gini",typ="regression",k=0)

# # forest, feature_gain, bootstrap_indices = random_forest_algorithm(X, y,bootstrap=True,n_trees=50,
# #                                                                   n_bootstrap=400, n_features=2,
# #                                                                   dt_max_depth=2,method="gini",typ="regression",k=0)
# # feature_gain_result = pd.DataFrame(columns=["tree_num","feature", "value"])

# # for i,j in enumerate(feature_gain):
# #     if j != -1:
# #         feature_gain_result.loc[i] = j
# #         if pd.api.types.is_list_like(j[2]) ==True:
# #             feature_gain_result.loc[i,"value"] = j[2][0]
            

# # y_predict = random_forest_predictions(X_test, forest)

# # mse = mean_squared_error(y_test,y_predict)
# # mse

# # fi_simulation = feature_gain_result.groupby(['tree_num','feature']).sum()
# # fi_simulation_s = fi_simulation.groupby(['feature']).sum()
# # fi_simulation_s.sort_values("value")

# # %% [markdown]
# # - k=tree_depth
# # - priority: 
# #     - k=1 vs k=0
# #     - benchmark:11 (real+synthetic)
# #     - feature_importance (order)
# #     - performance: mse (bootstrap(2) /test set(1) / cross validation)
# #     - fit model: sampled indices (for each tree)
# #     - out of bag prediction: unsampled indices (test set)

# # %%
# mse_k0_from_scratch_test

# # %%
# fi_k0_simulation_s

# # %% [markdown]
# # Y = 10sin(πX1X2) + 20(X3 − 0.05)^2 + 10X4 + 5X5 + ε
# # 
# # X3>X4>X1≈X2>X5

# # %%
# # mse_k0_train_mean = []
# # mse_k1_train_mean = []
# # mse_k0_test_mean = []
# # mse_k1_test_mean = []

# # for v in mse_k0_from_scratch_test.keys():
# #     # v is the list of grades for student k
# #     mse_k0_train_mean.append(np.mean(mse_k0_from_scratch_train[v]))
# #     mse_k1_train_mean.append(np.mean(mse_k1_from_scratch_train[v]))
# #     mse_k0_test_mean.append(np.mean(mse_k0_from_scratch_test[v]))
# #     mse_k1_test_mean.append(np.mean(mse_k1_from_scratch_test[v]))


# # %%


# # %%


# # %%


# # %%
# feature_gain_tree

# # %%
# def pred_tree(dtree, feature_gain_tree, coalition, row, node=0):
#     #left_node = clf.tree_.children_left[node]
#     #right_node = clf.tree_.children_right[node]
#     #is_leaf = left_node == right_node

#     #if is_leaf:
#     #    return decision_tree_predictions(row, dtree) ## change

#     feature = row.index[clf.tree_.feature[node]]
#     if feature in coalition:
#         if row.loc[feature] <= clf.tree_.threshold[node]:
#             # go left
#             return pred_tree(clf, coalition, row, node=left_node)
#         # go right
#         return pred_tree(clf, coalition, row, node=right_node)

#     # take weighted average of left and right
#     wl = clf.tree_.n_node_samples[left_node] / clf.tree_.n_node_samples[node]
#     wr = clf.tree_.n_node_samples[right_node] / clf.tree_.n_node_samples[node]
#     value = wl * pred_tree(clf, coalition, row, node=left_node)
#     value += wr * pred_tree(clf, coalition, row, node=right_node)
#     return value

# # %%
# # original
# def pred_tree(clf, coalition, row, node=0):
#     left_node = clf.tree_.children_left[node]
#     right_node = clf.tree_.children_right[node]
#     is_leaf = left_node == right_node

#     if is_leaf:
#         return clf.tree_.value[node].squeeze()

#     feature = row.index[clf.tree_.feature[node]]
#     if feature in coalition:
#         if row.loc[feature] <= clf.tree_.threshold[node]:
#             # go left
#             return pred_tree(clf, coalition, row, node=left_node)
#         # go right
#         return pred_tree(clf, coalition, row, node=right_node)

#     # take weighted average of left and right
#     wl = clf.tree_.n_node_samples[left_node] / clf.tree_.n_node_samples[node]
#     wr = clf.tree_.n_node_samples[right_node] / clf.tree_.n_node_samples[node]
#     value = wl * pred_tree(clf, coalition, row, node=left_node)
#     value += wr * pred_tree(clf, coalition, row, node=right_node)
#     return value


# # %%
# X[:1].T.squeeze().index[clf.tree_.feature[5]]

# # %%
# X[:1].T.squeeze()

# # %%
# pred_tree(clf, coalition=['x1', 'x2', 'x3'], row=X[:1].T.squeeze())

# # %%
# X[:1]

# # %%
# y[:1]

# # %%
# from sklearn.tree import DecisionTreeRegressor
# clf = DecisionTreeRegressor(random_state=888, max_depth = 2)

# clf.fit(X, y)

# # %%
# clf.tree_.value

# # %%
# clf.tree_.children_left

# # %%
# clf.tree_.children_right

# # %%
# n_nodes = clf.tree_.node_count
# children_left = clf.tree_.children_left
# children_right = clf.tree_.children_right
# feature = clf.tree_.feature
# threshold = clf.tree_.threshold

# node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
# while len(stack) > 0:
#     # `pop` ensures each node is only visited once
#     node_id, depth = stack.pop()
#     node_depth[node_id] = depth

#     # If the left and right child of a node is not the same we have a split
#     # node
#     is_split_node = children_left[node_id] != children_right[node_id]
#     # If a split node, append left and right children and depth to `stack`
#     # so we can loop through them
#     if is_split_node:
#         stack.append((children_left[node_id], depth + 1))
#         stack.append((children_right[node_id], depth + 1))
#     else:
#         is_leaves[node_id] = True

# print(
#     "The binary tree structure has {n} nodes and has "
#     "the following tree structure:\n".format(n=n_nodes)
# )
# for i in range(n_nodes):
#     if is_leaves[i]:
#         print(
#             "{space}node={node} is a leaf node.".format(
#                 space=node_depth[i] * "\t", node=i
#             )
#         )
#     else:
#         print(
#             "{space}node={node} is a split node: "
#             "go to node {left} if X[:, {feature}] <= {threshold} "
#             "else to node {right}.".format(
#                 space=node_depth[i] * "\t",
#                 node=i,
#                 left=children_left[i],
#                 feature=feature[i],
#                 threshold=threshold[i],
#                 right=children_right[i],
#             )
#         )

# # %%
# 0.25+0.218+0.213+0.216+0.22

# # %%
# from sklearn import tree
# tree.plot_tree(clf)

# # %%
# pred_tree(clf, coalition=['x1', 'x2', 'x3'], row=X[:1].T.squeeze())

# # %%
# import shap
# import tabulate
# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(X[:1])
# print(tabulate.tabulate(pd.DataFrame(
#     {'shap_value': shap_values.squeeze(),
#      'feature_value': X[:1].values.squeeze()}, index=X.columns),
#                         tablefmt="github", headers="keys"))

# # %%
# from itertools import combinations
# import scipy

# def make_value_function(clf, row, col):
#     def value(c):
#         marginal_gain = pred_tree(clf, c + [col], row) - pred_tree(clf, c, row)
#         num_coalitions = scipy.special.comb(len(row) - 1, len(c))
#         return marginal_gain / num_coalitions
#     return value

# def make_coalitions(row, col):
#     rest = [x for x in row.index if x != col]
#     for i in range(len(rest) + 1):
#         for x in combinations(rest, i):
#             yield list(x)

# def compute_shap(clf, row, col):
#     v = make_value_function(clf, row, col)
#     return sum([v(coal) / len(row) for coal in make_coalitions(row, col)])

# print(tabulate.tabulate(pd.DataFrame(
#     {'shap_value': shap_values.squeeze(),
#      'my_shap': [compute_shap(clf, X[:1].T.squeeze(), x) for x in X.columns],
#      'feature_value': X[:1].values.squeeze()
#     }, index=X.columns),
#                         tablefmt="github", headers="keys"))

# # %%


# # %%


# # %%


# # %%
# abs(ti_rs).mean(axis=0).sort_values()

# # %%
# abs(ti_rs).sum(axis=0).sort_values()

# # %% [markdown]
# # ----------end

# # %%
# # Depict one member tree as rule set
# forest[0]

# # %%
# # Calculate predictions
# predictions = random_forest_predictions(X, forest) 

# # %%
# # Check error rate
# error = np.mean(np.vstack(predictions) == np.array(y))

# error

# # %% [markdown]
# # The tree is able to classify a good proportion of the test set well even with only a depth of 3. 
# # 
# # Before moving on with using RF in `sklearn`, here is a **little exercise** for you:
# # Use your custom implementation of RF to develop a bagging ensemble.

# # %%
# # Your task: use the above functions to train a bagging ensemble using decision tress as base model
# def Bagging_algorithm(X, y, n_trees, n_bootstrap, n_features, dt_max_depth):
#     'puts the bootstrap sample in the decision tree algorithm with max depth and the random subset of features set, in otherwords, builds the forest tree by tree'
#     forest = []
#     for i in range(n_trees): #loops for the amount of trees set to be in the forest
#         X_bootstrapped, y_bootstrapped = bootstrapping(X, y, n_bootstrap)
#         tree = decision_tree_algorithm(X_bootstrapped, y_bootstrapped, max_depth=dt_max_depth, random_subspace=None) #creates individual trees
#         forest.append(tree)
    
#     return forest

# # %%
# # Grow bagging ensenble
# forest2 = Bagging_algorithm(X, y, n_trees=50, n_bootstrap=800, n_features=8, dt_max_depth=3)

# # %%
# # Depict one member tree as rule set
# forest2[0]

# # %%
# # Calculate predictions
# predictions2 = random_forest_predictions(X, forest2) 

# # %%
# # Check error rate
# error2 = np.mean(np.vstack(predictions2) == np.array(y))

# error2

# # %% [markdown]
# # Although the error2 (accuracy of bagging) is slightly bigger than error (accuracy of RF), the generationalized error of bagging might be greater than of RF

# # %% [markdown]
# # ## Random forest in sklearn
# # As promised, we will also demonstrate how you would use RF in production environments, that is with some proper implementation under the hood. Luckily, RF is maybe one of the most widely available data science algorithms. You find professional implementations on almost every platform. Obviously, this also includes `sklearn`. 
# # 
# # It turns out that training and using RF in `sklearn`is super easy. Check out this code, which shows all it takes.

# # %%
# from sklearn.ensemble import RandomForestClassifier  # import library
# rf = RandomForestClassifier()                        # create model object
# rf.fit(X_train, y_train)                             # fit model to training set 
# yhat = rf.predict_proba(X_test)                      # obtain test set predictions                   

# # %% [markdown]
# # And that was it. We promised it is easy, did we not? ;)
# # 
# # However, the above demo is simplistic and does not really illustrate how you would use RF in practice. That is mainly because we omit hyperparameters and their tuning. **Using an analytical model with some default parameters is never a good idea.** 

# # %% [markdown]
# # #### Tuning RF hyperparameters using grid search
# # Random forest is often considered robust regardless of hyperparameters settings. Still, some tuning may be beneficial. For model selection, we draw on the demos of [Tutorial 7](https://github.com/Humboldt-WI/bads/blob/master/tutorials/7_nb_model_selection.ipynb) and adjust these for our focal task of tuning RF. 
# # 
# # We consider the following hyperparameters. 
# # <br>
# # - `n_estimators`: number of trees(models) in forest (ensemble)<br>
# # - `max_features` : maximum features in random subspace<br>
# # 
# # There are a couple more hyperparameters. Normally, you would not need to tune these but for the sake of completeness, here are some more hyperparameters:
# # - `min_samples_split`: minimum number of samples required in leaf node before another split is made. If below, node won't be split.<br>
# # - `min_samples_leaf`: minimum number of samples required to be in a leaf node.<br>
# # - `max_leaf_nodes`: maximum number of leaf nodes in a tree<br>
# # - `criterion`: splitting function to use, e.g. gini coefficient<br>
# # - `max_depth`: pruning parameter, maximum depth of decision tree<br>
# # - `n_jobs`: parallelization of model building<br>
# # - `random_state`: sets 'pattern' of random selection for reproducibility <br><br>
# # 
# # If hyperparameters are not specified, they will be set to their default. 
# # 
# # **Remark**: Tuning RF might take a while. If you want to speed things up, **try not to reduce `n_estimators` (number of trees).** Instead, consider setting the hyperparameter `max_samples`. It allows you to control the size of the bootstrap sample from which each tree is grown. Read the documentation for more information. Smaller sample sizes accelerate the training.

# # %%
# ?RandomForestClassifier

# # %%
# from sklearn.model_selection import GridSearchCV
# print('Tuning random forest classifier')
# rf = RandomForestClassifier(random_state=888, max_samples = 0.5)  # This way, bootstrap sample size will be 50% of the training set

# # Define meta-parameter grid of candidate settings
# # The following settings are just for illustration
# param_grid = {'n_estimators': [10, 20, 50],  # very small values and not suitable for application. Use larger sizes to build good classifiers
#               'max_features': [1, 2, 4]
#               }

# # Set up the grid object specifying the tuning options
# gs_rf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', verbose=1)
# gs_rf.fit(X_train, np.ravel(y_train))

# # %% [markdown]
# # verboseint
# # Controls the verbosity: the higher, the more messages.
# # 
# # >1 : the computation time for each fold and parameter candidate is displayed;
# # 
# # >2 : the score is also displayed;
# # 
# # >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.

# # %% [markdown]
# # Let's say we are interested in assessing our RF model in terms of ROC analysis. 

# # %%
# print("Best CV AUC: %0.4f" % gs_rf.best_score_)
# print("Optimal RF meta-parameters:")
# print(gs_rf.best_params_)

# from sklearn import metrics

# auc_trace = {}

# # Find test set auc of the best random forest classifier
# fp_rate, tp_rate, _ = metrics.roc_curve(y_test, gs_rf.predict_proba(X_test)[:, 1])
# auc_trace.update( {'rf' : metrics.auc(fp_rate, tp_rate)}) 
# print('RF test set AUC: {:.4f}'.format(auc_trace['rf']))

# # %% [markdown]
# # You should see some quite impressive AUC value. Let's plot the ROC curve to appreciate the power of our RF. This also shows how to access the final model from the grid-search results.

# # %%
# # The plot is not new but note the use of gs_rf.best_estimator_ 
# metrics.plot_roc_curve(gs_rf.best_estimator_, X_test, y_test)
# plt.plot([0, 1], [0, 1], "r--");

# # %% [markdown]
# # # Boosting
# # Boosting-type algorithms are based on **two core principles**:
# # 1. Develop an ensemble sequentially (i.e. add one model at a time)
# # 2. Let the next model in the chain correct the errors of the current ensemble
# # 
# # We discussed two flavors of boosting in the lecture:
# # - **adaboost algorithm**, first instantiation of a boosting ensemble
# # - **gradient boosting**, proposed in the paper [Greedy Function Approximation: A Gradient Boosting Machine](https://www.jstor.org/stable/2699986?seq=1) by Jerome H. Friedman
# # 
# # This part of the tutorial follows the same approach as above, we first demonstrate implementing a boosting ensemble from scratch and then showcase its use together with a professional machine learning library. Many derivatives of boosting and libraries exist. Some members of the gradient boosting family include:
# # - [LightGBM from Microsoft](https://lightgbm.readthedocs.io/en/latest/)
# # - [Catboost from Yandex](https://catboost.ai/)
# # - [NGBoost from Stanford ML Group](https://stanfordmlgroup.github.io/projects/ngboost/)
# # 
# # Given wide adoption in practice and academia, we opted for focusing the tutorial to **extreme gradient boosting** or XGB for short. Demos for other popular boosting algorithms are available via the links above. Boosting algorithms that do not follow the gradient boosting principle may be considered somewhat old-fashioned. We will not cover them here. However, the exercises give you an opportunity to implement your own Adaboost ensemble from scratch. And of course, `sklearn` also supports adaboost and other boosting algorithms.    

# # %% [markdown]
# # ## Verifying the boosting principle
# # Is it true really true that one model can *learn* or *correct* the errors of another model as promised by the boosting paradigm? Before diving into cutting-edge gradient boosting, let's convince ourselves of this  fundamental premise of boosting. 
# # 
# # For this purpose, we use the HMEQ data set and demonstrate that training a model on the errors of a previous model helps to reduce classification error. Since it is common practice to implement boosting using trees as base model, we follow this approach.  

# # %%
# from sklearn import tree

# # %% [markdown]
# # ### Model training 
# # Here we will show the effectiveness of corrective models that *train on errors*. We will first train two models, the first will be for regular predictions. The second will predict which observations the first model misclassifies. We will first run the first prediction on test data, then correct these predictions using the second model.
# # 

# # %%
# estimators = []   # List to store the different models

# # %%
# # Train first classifier
# clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=2, max_depth=2)  # First classifier
# dt = clf.fit(X_train, y_train)  # Fit the classifier
# estimators.append(('first model', dt))

# # %% [markdown]
# # Having obtained our first model, we can calculate that model's residuals on the training set. Let's consider a classification setting. More concretely, let's focus on discrete class predictions. Recall that this is the type of output that we obtain when calling the function `predict()`.  

# # %%
# initial_pred = dt.predict(X_train)  # Classify training set using first classifier

# # %% [markdown]
# # Next, we identify misclassified observations and calculate the **classification error**.
# # 
# # Let's produce a binary vector with the same length as the training set in which an entry of `True` indicates that the corresponding training set observation is misclassified by our first tree. The way we design our vector also makes it very easy to compute the classification error. All we need is to find the mean of that vector.

# # %%
# res = initial_pred != y_train.iloc[:, 0] 
# print("Classification error of tree #1 is {:.4}".format(res.mean()))
# print("Total number of errors tree #1 is", res.sum())

# # %% [markdown]
# # On to model #2. Here, we train not on $y$ but a new binary target variable indicating whether model #1 classified an observation correctly. We can think of this new target as $y - \hat{y}$ ie. the residual. However, since we train on decisions of a binary outcome, this classifier will predict errors of the first classifier.

# # %%
# clf2 = tree.DecisionTreeClassifier(criterion="gini")  # instantiate second model
# dt_res = clf2.fit(X_train, res)                       # note the new target, we are training on residuals
# estimators.append(('second model', dt_res))           # store second model for later
# dt_res

# # %% [markdown]
# # Calling the `predict`function of our second corrective model, we obtain a prediction of which observations model #1 is likely to misclassify. 

# # %%
# likely_misclassifications = dt_res.predict(X_train) 
# print("Based on model #2, we expect model #1 to misclassify {} observations.".format(
#     likely_misclassifications.sum()))

# # %%
# # Check if classified likely misclassifications are the same as residuals
# accuracy_misclassifications = likely_misclassifications == res
# print(accuracy_misclassifications)

# # %%
# # It seems likely misclassifications are exactly congruent with residuals, so the model does work
# accuracy_misclassifications.mean() 

# # %% [markdown]
# # ### Model testing
# # 
# # Now that we have our two models, we will begin using the test data to see if a combination of the two models can reduce the value of the residuals. We will first predict y using X_test.

# # %%
# pred_initial_test = dt.predict(X_test)
# print(pred_initial_test.shape)
# pred_initial_test

# # %%
# res_test = pred_initial_test != y_test.iloc[:, 0]
# print("Test error of model 1: {:.4}".format(res_test.mean()))

# # %% [markdown]
# # Now we predict for which observations model 1 has likely made an error.

# # %%
# likely_misclassifications_test = dt_res.predict(X_test)
# print(likely_misclassifications_test.shape)
# print(likely_misclassifications_test.sum())
# likely_misclassifications_test

# # %% [markdown]
# # Lastly, we correct the likely misclassifications by simply flipping the predicted (from model 1) class label.

# # %%
# pred_corrected = pd.Series(pred_initial_test)
# pred_corrected

# # %%
# pred_corrected = pd.Series(pred_initial_test)
# pred_corrected[likely_misclassifications_test] = ~ pred_corrected[likely_misclassifications_test]
# pred_corrected

# # %%
# print(pred_initial_test[likely_misclassifications_test].shape)  # Check that they have actually been changed
# pred_initial_test[likely_misclassifications_test]

# # %%
# pred_corrected[likely_misclassifications_test]  # all the results are opposite, so this worked!

# # %% [markdown]
# # Time for the grand final, did we reduce the test error?

# # %%
# res_corrected = np.array(pred_corrected) != y_test.iloc[:,0]
# print("Test error after corrected model 1 by model 2: {:.4}".format(res_corrected.mean()))

# # %% [markdown]
# # Hurray!!!
# # 
# # A lower test error indicates that our process worked. We were able to lower the error on a test set using a second model which focused on identifying misclassified cases. Let's now examine how gradient boosting relies on similar principles but is a bit more complex in execution.

# # %% [markdown]
# # ## Gradient Boosting from scratch
# # 
# # Gradient boosting is one specific form of boosting (using residuals recursively to increase accuracy). This process begins with an initial simple prediction, which is often the mean of the target variable. Next, the algorithm iteratively goes through every feature and determines which feature will best reduce this error with a single split. This is essentially a single level tree, or stump. This stump is then chosen and added to the ensemble. Next, residuals are calculated once again and the process continues for as many iterations as deemed necessary.
# # 
# # We can implement this procedure from scratch as well to examine how it works. The original code for this exercise can be seen [here](https://towardsdatascience.com/gradient-boosting-in-python-from-scratch-4a3d9077367). It has been adapted for this lesson.  We will first use the same data set that we generated in [Tutorial 3 for classification](https://github.com/Humboldt-WI/bads/blob/master/tutorials/3_nb_predictive_analytics.ipynb). Our gradient booster will also be a classification example. Keep in mind that regression is possible too!

# # %%
# np.random.seed(888)

# # %%
# # Create synthetic dataset, same as tutorial 3

# class1_x = np.random.normal(loc=1, scale=1, size=1000)
# class1_y = np.random.normal(loc=1, scale=1, size=1000)

# class2_x = np.random.normal(loc=4, scale=1, size=1000)
# class2_y = np.random.normal(loc=4, scale=1, size=1000)

# lab1 = np.repeat(0, 1000)
# lab2 = np.repeat(1, 1000)

# class1 = np.vstack((class1_x, class1_y)).T
# class2 = np.vstack((class2_x, class2_y)).T

# data = np.vstack((class1,class2))

# labels = np.concatenate((lab1,lab2))

# # Visualization of data set

# plt.scatter(data[:,0], data[:,1], c=labels, alpha=.3);
# plt.xlabel("$x_1$");
# plt.ylabel("$x_2$");

# # %% [markdown]
# # To begin gradient boost, we first need a loss function to quantify the difference between the predicted y values and the true y values. This loss function must be differentiable to obtain the gradient. This loss function will determine how the function and its derivative (the gradient) creates future stumps for our gradient boost forest.

# # %%
# def compute_loss(y, pred_y): 
#     return ((y - pred_y) ** 2) / 2

# # %%
# def loss_gradient(y, pred_y): 
#     return -(y-pred_y)

# # %% [markdown]
# # Now that we have our loss and gradient functions, we can begin with the gradient boost algorithm. The first step in the algorithm is to create an initial prediction. Generally, this is the mean of the labels. Next, residuals are calculated by using the `loss_gradient` function and a shallow single level decision tree or decision 'stump' is created based on these residuals. Residuals are then recalculated and this process repeats iteratively.

# # %%
# from sklearn.tree import DecisionTreeRegressor 

# def gradient_boost_algorithm(X, y, iter=10): 
#     forest_of_stumps = [] 
#     pred_y = np.array([y.mean()]*len(y))  # Predict mean for all observations
#     pred_y_initial = pred_y 
#     print(compute_loss(y, pred_y).mean()) 
#     for i in range(iter): 
#         residuals = -loss_gradient(y, pred_y)  # Calculate residuals using the gradient function
#         stump = DecisionTreeRegressor(max_depth=1)
#         stump.fit(X, residuals)  # Feed data and residuals to the decision tree function (stump since it has a depth of only 1)
#         forest_of_stumps.append(stump)  # Add this tree to the forest
#         new_pred_y = stump.predict(X)  # Make new predictions and repeat this process over the specified number of iterations
#         pred_y = pred_y + new_pred_y  # note that this update shares similarities with the gradient descent algorithm, just that our basic implementation does not use a learning rate parameter
#         print(compute_loss(y, pred_y).mean())
#     return forest_of_stumps, pred_y_initial

# # %% [markdown]
# # The final function will be to predict classes using our algorithm. We will first run new observations through our forest of stumps sequentially and add their predictions together. At this point, the tree will return probabilities that each observation will be of class `1`. If we want a class prediction rather than probabilities of class `1`, we can compare each prediction to some threshold value (defaulted at `0.5`). Predictions over the threshold will be considered class `1` predictions while all others will be class `0`.

# # %%
# def gradient_boost_predict(forest_of_stumps, pred_y_initial, X, predict_class=False, threshold=0.5): 
#     pred_y = np.array([pred_y_initial[0]]*len(X))
#     for stump in forest_of_stumps: 
#         pred_y = pred_y + stump.predict(X)  # Navigate the forest sequentially adding the predictions from each stump along the way
#     if predict_class:
#       pred_y = pred_y > threshold
#       pred_y = np.where(pred_y==False, 0, pred_y)
#       pred_y = np.where(pred_y==True, 1, pred_y) 
#     return pred_y

# # %% [markdown]
# # Now we can train our model using these functions. Each loop within the function will print the mean of the loss function so we can verify that it is actually decreasing. You can adjust iteration number to allow a larger forest of stumps in the gradient boost forest.

# # %%
# stumps, pred_y_initial = gradient_boost_algorithm(data, labels, iter=10)

# # %%
# stumps

# # %% [markdown]
# # We can now use these stumps to make a prediction on the data.

# # %%
# pred_gradient_boost = gradient_boost_predict(stumps, pred_y_initial, data, predict_class=True)

# print(pred_gradient_boost)

# # %% [markdown]
# # We can also assess the mean error on the labels which is very low. Therefore, we can see this method works very well to separate classes.

# # %%
# res_gradient_boost = pred_gradient_boost != labels
# print("The average error is", res_gradient_boost.mean())
# print("The total number of errors is" , res_gradient_boost.sum(), "of", len(res_gradient_boost), "predictions in total")

# # %% [markdown]
# # ## XGBoost with xgb library 
# # Different gradient boosting algorithms exist and different implementations exist of each of those. If you want to use the *original gradient boosting machine* as proposed in [Friedman's 2001 paper](https://www.jstor.org/stable/2699986?seq=1) or his [follow-up paper on stochastic gradient boosting](http://dx.doi.org/10.1016/S0167-9473(01)00065-2), we recommend using the class `sklearn.ensemble.GradientBoostingClassifier`. It offers a lot of flexibility (e.g., hyperparameters) and is[ very well documented](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html). However, for lager data sets, you might want to use **extreme gradient boosting (XGB)**, which was explicitly designed for highly scalable gradient boosting. Below, we demonstrate the application of XGB to our credit risk data set.
# # 
# # Although `sklearn` does have a version of this algorithm, it is actually common to use the `xgboost` library for training XGB models. To be precise, `sklearn` offers an implementation of XGB but it might not incorporate the latest features. As far as the use is concerned, you will not notice any differences between the `xgboost` library and `sklearn`. However, **you might need to install the `xgboost` library before moving on**.

# # %% [markdown]
# # ### Tuning XGB hyperparameters
# # Our modeling pipeline is exactly as before in the RF example, and in general. We tune hyperparameters using grid-search and cross-validating the training data. Once we determined good hyperparameter settings, we train a XGB model with the corresponding configuration on the entire training set and obtain test set prediction, which we assess using ROC analysis. 

# # %%
# #pip install xgboost


# # %%
# import xgboost as xgb
#     # Setting up the grid of meta-parameters
# xgb_param_grid = {
#     'colsample_bytree': np.linspace(0.5, 0.9, 5),  # random subspace
#     'n_estimators': [10, 20],  # ensemble size or number of gradient steps; set to rather small values to save time
#     'max_depth': [5, 10],   # max depth of decision trees
#     'learning_rate': [0.1, 0.01],  # learning rate
#     'early_stopping_rounds': [10]}  # early stopping if no improvement after that many iterations

# gs_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=xgb_param_grid, scoring='roc_auc', cv=5, verbose=0)

# gs_xgb.fit(X_train, np.ravel(y_train))

# # %%
# print("Best CV AUC: %0.4f" % gs_xgb.best_score_)
# print("Optimal XGB meta-parameters:")
# print(gs_xgb.best_params_)

# # Find test set AUC of the best XGB classifier
# fp_rate, tp_rate, _ = metrics.roc_curve(y_test, gs_xgb.predict_proba(X_test)[:, 1])
# print('XGB test set AUC with optimal meta-parameters: {:.4f}'.format(metrics.auc(fp_rate, tp_rate) ))

# # %%
# # The plot is not new but note the use of gs_xgb.best_estimator_ 
# metrics.plot_roc_curve(gs_xgb.best_estimator_, X_test, y_test)
# plt.plot([0, 1], [0, 1], "r--");

# # %% [markdown]
# # # Conclusions
# # With our simple credit risk data, RF and XGB both predict extremely accurately and it is hard to draw conclusions related to which one does better. Many recent studies and competitions find XGB to outperform RF provided you carefully tune XGB hyperparameters. Either may, you have seen some of the most popular off-the-shelf prediction methods in action and went through the burden of coding these methods from scratch. That is a remarkable accomplishment! Time put pad yourself on the shoulder and conclude our journey through the space of ensemble learning algorithms. 

# # %%



