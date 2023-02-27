import numpy as np
import pandas as pd
from Data_Prep import load_data, data_pipeline, test_model


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

import warnings

'''
in Model_Selection.py, it was determined that the tree-based models performed
the best, which are Gradient-Boosted Trees, Random Forests, and Decision Trees

Gradient-Boosted Trees and Random Forests had the highest f1-scores and Precision,
while decision tree had the highest recall (excluding naive bayes which performed
terribly everywhere else)

Therefore, I am going to tune the tree-based models, as well as logistic regression,
just so I'm tuning something a little different
'''

def tune_logreg(X, y):
    lr = LogisticRegression()
    
    logistic_param_dict = {'penalty':['none' , 'l2', 'l1'],
                           'class_weight': [{0:1, 1:1},
                                            {0:0.5, 1:1.5},
                                            {0:0.1, 1:1.9}],
                           'solver': ['lbfgs', 'liblinear', 'newton-cholesky']}
    lr_grid_search = GridSearchCV(lr, logistic_param_dict, scoring='f1',
                                  refit=False, error_score=0.01, cv=3,
                                  verbose=1)
    lr_grid_search.fit(X, y)
    return lr_grid_search

def tune_dectree(X, y):
    dtc = DecisionTreeClassifier()
   
    tree_param_dict = {'criterion': ['gini', 'entropy', 'log_loss'],
                       'max_depth': [5, 10, 15, 20, 50, 100],
                       'min_samples_split': [1, 2, 5],
                       'max_features': [10, 'sqrt', 'log2', 'None'],
                       'max_leaf_nodes': [10, 20, 50, 100, 200],
                       'class_weight': [{0:1, 1:1},
                                        {0:0.5, 1:1.5},
                                        {0:0.1, 1:1.9}],}
    
    dtc_grid_search = GridSearchCV(dtc, tree_param_dict, scoring='f1',
                                   refit=False, error_score=0.01, cv=10,
                                   verbose=2)
   
    dtc_grid_search.fit(X, y)
    return dtc_grid_search

def tune_randforest(X, y):
    rfc = RandomForestClassifier()
    
    forest_param_dict = {'n_estimators': [60, 100, 140],
                         'criterion': ['gini', 'entropy', 'log_loss'],
                         'max_depth': [50, 100],
                         'min_samples_split': [2, 5],
                         'max_features': [10, 'None'],
                         'max_leaf_nodes': [20, 50],
                         'class_weight': [{0:1, 1:1},
                                          {0:0.5, 1:0.5}],
                         'max_samples': [10, 50, 100, 200]}
    
    forest_gridsearch = GridSearchCV(rfc, forest_param_dict, scoring='f1',
                                     refit=False, error_score=0.01, cv=3,
                                     verbose=2)
    forest_gridsearch.fit(X, y)
    return forest_gridsearch

def tune_gradientboost(X, y):
    gbt = GradientBoostingClassifier()
    
    gbt_params = {'loss': ['log_loss', 'exponential'],
                  'learning_rate': [0.05, 0.1],
                  'criterion': ['friedman_mse', 'squared_error'],
                  'max_depth': [50, 100],
                  'min_samples_split': [2, 5],
                  'max_features': [10, None],
                  'max_leaf_nodes': [20, 50],
                  }
    
    gbt_gridsearch = GridSearchCV(gbt, gbt_params, scoring='f1',
                                  refit=False, error_score=0.01, cv=3,
                                  verbose=2)
    gbt_gridsearch.fit(X, y)
    return gbt_gridsearch


### logistic regression:
# class weights: {0: 0.5, 1:1.5}
# penalty: l1
# solver: liblinear

# f1 score: 0.550 -> 0.630


### decision tree:
# class weights: {0: 1, 1:1}
# criterion: gini
# max_depth: 100
# max_features: 10
# max_leaf_nodes: 20
# min_samples_split: 5

# f1 score: 0.658 -> 0.715


### random forest:
# class_weight: {0: 1, 1: 1}
# criterion: gini
# max_depth: 100
# max_features: 10
# max_leaf_nodes: 50
# max_samples: 200
# min_samples_split: 2
# n_estimators: 60

# f1 score: 0.706 -> 0.732


### gradient boosted tree:
# criterion: friedman_mse
# learning_rate: 0.05
# loss: exponential
# max_depth: 100
# max_features: 10
# max_leaf_nodes: 50
# min_samples_split: 5

# f1 score: 0.700 -> 0.733 

def eval():
    #### validating models on validation data
    # for a second I was worried about data leakage, since the validation-normal split
    # re-randomizes each time, but with the test_model method, the models are re-fit
    
    pokemon_df = load_data()
    X, X_valid, y, y_valid = data_pipeline(pokemon_df, drop_prevos=True,
                                           kind='minmax', valid_size=0.1)
    
    #logistic regression - 0.630:
    lr = LogisticRegression(class_weight={0:0.5, 1:1.5}, penalty='l1',
                            solver='liblinear')
    test_model(lr, X, y, print_info=True)
    
    #decision tree - 0.640:
    dtc = DecisionTreeClassifier(class_weight={0:1, 1:1}, criterion='gini',
                                 max_depth=100, max_features=10, 
                                 max_leaf_nodes=20, min_samples_split=5)
    test_model(dtc, X, y, print_info=True)
    
    #random forest - 0.712:
    rfc = RandomForestClassifier(class_weight={0:1, 1:1}, criterion='gini', 
                                 max_depth=100, max_features=10,
                                 max_leaf_nodes=20, max_samples=200,
                                 min_samples_split=2, n_estimators=60)
    test_model(rfc, X, y, print_info=True)
    
    # gradient boosted tree - 0.722:
    gbt = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.05,
                                     loss='exponential', max_depth=100,
                                     max_features=10, max_leaf_nodes=50,
                                     min_samples_split=5)
    test_model(gbt, X, y, print_info=True)
    
    # gradient boosted tree wins: pretty much all of its metrics are higher than
    # the other models' metrics. Thinking back, due to the amount of grid search,
    # it's likely that the maximum values were outliers. Anyways, here's the validation 
    # for the same models:
        
    print(f1_score(lr.predict(X_valid), y_valid)) # 0.614
    print(f1_score(dtc.predict(X_valid), y_valid)) # 0.700
    print(f1_score(rfc.predict(X_valid), y_valid)) # 0.737
    print(f1_score(gbt.predict(X_valid), y_valid)) # 0.5
    
    #results remain somewhat consistent across repeated runs
    
    # this is incredibly surprising. In nearly every case, gradient boosted trees
    # beat out the other models in the testing phase. But in the validation phase,
    # the gradient boosted trees no longer outperforms. Random Forest
    # seems to be the best option here, as its the most resistant to overfitting.

def main():
    warnings.simplefilter("ignore")
    
    lrlist = []
    dtclist = []
    rfclist = []
    gbtlist = []
    
    for i in range(10):
        pokemon_df = load_data()
        X, X_valid, y, y_valid = data_pipeline(pokemon_df, drop_prevos=True,
                                           kind='minmax', valid_size=0.1)
        
        lr = LogisticRegression(class_weight={0:0.5, 1:1.5}, penalty='l1',
                                solver='liblinear')
        test_model(lr, X, y)
        lrlist.append(f1_score(lr.predict(X_valid), y_valid))
        
        dtc = DecisionTreeClassifier(class_weight={0:1, 1:1}, criterion='gini',
                                     max_depth=100, max_features=10, 
                                     max_leaf_nodes=20, min_samples_split=5)
        test_model(dtc, X, y)
        dtclist.append(f1_score(dtc.predict(X_valid), y_valid))
        
        rfc = RandomForestClassifier(class_weight={0:1, 1:1}, criterion='gini', 
                                     max_depth=100, max_features=10,
                                     max_leaf_nodes=20, max_samples=200,
                                     min_samples_split=2, n_estimators=60)
        test_model(rfc, X, y)
        rfclist.append(f1_score(rfc.predict(X_valid), y_valid))
        
        gbt = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.05,
                                         loss='exponential', max_depth=100,
                                         max_features=10, max_leaf_nodes=50,
                                         min_samples_split=5)
        test_model(gbt, X, y)
        gbtlist.append(f1_score(gbt.predict(X_valid), y_valid))
        print('sample', i, 'done')
    
    print(f'LOGREG:\tMEAN:{np.mean(lrlist)}\tSTD:{np.std(lrlist)}')
    print(f'DECTREE:\tMEAN:{np.mean(dtclist)}\tSTD:{np.std(dtclist)}')
    print(f'RANDFOREST:\tMEAN:{np.mean(rfclist)}\tSTD:{np.std(rfclist)}')
    print(f'GRADBOOST:\tMEAN:{np.mean(gbtlist)}\tSTD:{np.std(gbtlist)}')
    
    #upon further testing and analysis, the choice of random forest has still remained
    # it has consistently score ever so slightly higher than the other models
    # on the validation sets. Random forest it is.
    
    
if __name__ == '__main__':
    main()