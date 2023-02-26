#modelling, cross-validation and viewing results


import numpy as np
import pandas as pd
from Data_Prep import load_data, data_pipeline, test_model


def model_experimenting():
    #import_data
    pokemon_df = load_data()
    X, X_valid, y, y_valid = data_pipeline(pokemon_df, drop_prevos=True,
                                           kind='minmax', valid_size=0.1)
    
    metric_list = []
    ####LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    logreg_list = test_model(lr, X, y)
    metric_list.append(logreg_list)
    
    # did slightly better with standard scaling
    
    ####DECISION TREE
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dectree_list = test_model(dt, X, y)
    metric_list.append(dectree_list)
    
    # did slightly worse with standard scaling
    
    ####NAIVE BAYES
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    bayes_list = test_model(gnb, X, y)
    metric_list.append(bayes_list)
    
    ####RANDOM FOREST
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    randforest_list = test_model(rfc, X, y)
    metric_list.append(randforest_list)
    
    # did around equivalent with standard scaling
    
    ####NEAREST NEIGHBORS
    from sklearn.neighbors import KNeighborsClassifier
    knc = KNeighborsClassifier()
    nn_list = test_model(knc, X, y)
    metric_list.append(nn_list)
    
    # did significantly better with standard scaling
    
    ####SVM
    from sklearn.svm import SVC
    mysvm = SVC()
    svm_list = test_model(mysvm, X, y)
    metric_list.append(svm_list)
    
    
    ####Gradient-Boosted Decision tree
    from sklearn.ensemble import GradientBoostingClassifier
    gbt = GradientBoostingClassifier()
    gbt_list = test_model(gbt, X, y)
    metric_list.append(gbt_list)
    
    metrics_df = pd.DataFrame(metric_list).T
    metrics_df.columns = ['logreg', 'dectree', 'naivebayes', 'randforest', 'nn',
                          'svm', 'gbtree']
    print(metrics_df)
    #standouts seem to be randforest, gbtree. naivebayes had a very high recall,
    # but everything else was really bad
    # i can't think of a reason why dectree would have a higher recall, but that's
    # probably because of no tuning

def main():
    model_experimenting()

if __name__ == '__main__':
    main()