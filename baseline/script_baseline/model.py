import numpy as np
import sklearn
from sklearn import linear_model,tree
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor,\
GradientBoostingRegressor,BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb





AdaBoost = AdaBoostRegressor(base_estimator=sklearn.ensemble.ExtraTreesRegressor(n_jobs=60), 
                       n_estimators=10,
                       learning_rate=1.0, 
                       loss='linear', 
                       random_state=None)

Bagging = BaggingRegressor(base_estimator=None, 
                       n_estimators=100, 
                       max_samples=100, 
                       max_features=1.0, 
                       bootstrap=True, 
                       bootstrap_features=False, 
                       oob_score=False, 
                       warm_start=False, 
                       n_jobs=60,
                       random_state=None, 
                       verbose=0)

DecisionTree = tree.DecisionTreeRegressor(criterion='squared_error', 
                       splitter='best', 
                       max_depth=None, 
                       min_samples_split=2, 
                       min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, 
                       max_features=None, 
                       random_state=None, 
                       max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, 
                       ccp_alpha=0.0)
  
    
ExtraTrees = ExtraTreesRegressor(n_estimators=60, 
                       criterion='squared_error', 
                       max_depth=None, 
                       min_samples_split=2,
                       min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, 
                       max_features='auto', 
                       max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, 
                       bootstrap=False, 
                       oob_score=False, 
                       n_jobs=60, 
                       random_state=None, 
                       verbose=0, 
                       warm_start=False, 
                       ccp_alpha=0.0, 
                       max_samples=None)    
GradientBoosting=GradientBoostingRegressor(loss='ls', 
                       learning_rate=0.1, 
                       n_estimators=100, 
                       subsample=1.0, 
                       criterion='friedman_mse', 
                       min_samples_split=2, 
                       min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, 
                       max_depth=4, 
                       min_impurity_decrease=0.0, 
                       init=None, 
                       random_state=None, 
                       max_features=None, 
                       alpha=0.9, 
                       verbose=0, 
                       max_leaf_nodes=None, 
                       warm_start=False, 
                       validation_fraction=0.1, 
                       n_iter_no_change=None, 
                       tol=0.0001, 
                       ccp_alpha=0.0)                          
KNeighbors=KNeighborsRegressor(n_neighbors=10,
                       weights='uniform',
                       algorithm='auto',
                       leaf_size=30,
                       p=2,
                       metric='minkowski',
                       metric_params=None,
                       n_jobs=None) 

KernelRidge = KernelRidge(alpha=1, 
                       kernel='linear', 
                       gamma=None, 
                       degree=3, 
                       coef0=1, 
                       kernel_params=None)

LinearSVR=LinearSVR(epsilon=0.0,
                       tol=0.0001,
                       C=1.0,
                       loss='epsilon_insensitive',
                       fit_intercept=True,
                       intercept_scaling=1.0,
                       dual=True,
                       verbose=0,
                       random_state=None,
                       max_iter=10000)

RandomForest =  RandomForestRegressor(n_estimators=100,
                       criterion='mae',
                       max_depth=None,
                       min_samples_split=2,
                       min_samples_leaf=1,
                       min_weight_fraction_leaf=0.0,
                       max_features='auto',
                       max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       bootstrap=True,
                       oob_score=False,
                       n_jobs=60,
                       random_state=None,
                       verbose=0,
                       warm_start=False,
                       ccp_alpha=0.0,
                       max_samples=None)

Ridge = linear_model.Ridge(alpha=.5,
                       fit_intercept=True,
                       #normalize=False,
                       copy_X=True,
                       max_iter=None,
                       tol=0.001,)

SVR = SVR(kernel='rbf', 
                      degree=3, 
                      gamma='scale', 
                      coef0=0.0, 
                      tol=0.001, 
                      C=1.0, 
                      epsilon=0.1, 
                      shrinking=True, 
                      cache_size=200, 
                      verbose=False, 
                      max_iter=- 1)
XGB = xgb.XGBRegressor(base_score=0.5, 
                      booster='gbtree', 
                      colsample_bylevel=1, 
                      colsample_bynode=1, 
                      colsample_bytree=1, 
                      gamma=0, 
                      gpu_id=-1, 
                      importance_type='gain',
                      interaction_constraints='', 
                      learning_rate=0.3, 
                      max_delta_step=0, 
                      max_depth=10, 
                      min_child_weight=1, 
                      missing=np.nan, 
                      monotone_constraints='()', 
                      n_estimators=60, 
                      num_parallel_tree=1, 
                      random_state=0, 
                      reg_alpha=0, 
                      reg_lambda=1, 
                      scale_pos_weight=1, 
                      subsample=1, 
                      tree_method='exact', 
                      validate_parameters=1, 
                      verbosity=None)                          
NeuralNetwork = MLPRegressor(hidden_layer_sizes=(100,100), 
                      activation='relu', 
                      solver='adam', 
                      alpha=0.0001, 
                      batch_size='auto', 
                      learning_rate='constant', 
                      learning_rate_init=0.001, 
                      power_t=0.5, 
                      max_iter=200, 
                      shuffle=True, 
                      random_state=None, 
                      tol=0.0001, 
                      verbose=False, 
                      warm_start=False, 
                      momentum=0.9, 
                      nesterovs_momentum=True, 
                      early_stopping=False, 
                      validation_fraction=0.1, 
                      beta_1=0.9, 
                      beta_2=0.999, 
                      epsilon=1e-08, 
                      n_iter_no_change=10, 
                      max_fun=15000)                          
                                               
                          
models = [AdaBoost,Bagging,
          DecisionTree,ExtraTrees,GradientBoosting,
          KNeighbors,KernelRidge,
          LinearSVR,RandomForest,
          Ridge,SVR, XGB,
          NeuralNetwork]
                          
                          
'''models = [AdaBoostRegressor(ExtraTreesRegressor(n_jobs=60)),BaggingRegressor(n_jobs=60),
          tree.DecisionTreeRegressor(),ExtraTreesRegressor(n_jobs=60),GradientBoostingRegressor(),
          KNeighborsRegressor(),KernelRidge(),
          LinearSVR(),RandomForestRegressor(n_jobs=60,criterion='mae',n_estimators=10,max_depth=10),
          linear_model.Ridge(alpha=.5),SVR(), xgb.XGBRegressor(n_jobs=60),
          MLPRegressor(hidden_layer_sizes=(100,100))]'''