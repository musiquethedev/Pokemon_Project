import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Data_Prep import data_pipeline
from joblib import dump


def create_model():
    pokemon_random_forest = RandomForestClassifier(class_weight={0:1, 1:1},
                                                   criterion='gini', 
                                                   max_depth=100,
                                                   max_features=10,
                                                   max_leaf_nodes=20,
                                                   max_samples=200,
                                                   min_samples_split=2,
                                                   n_estimators=60)
    pokemon_df = pd.read_csv("Natdex_Data.csv")
    X, y = data_pipeline(pokemon_df, full_data=True)
    pokemon_random_forest.fit(X, y)
    return pokemon_random_forest

def save_model(df):
    dump(df, 'Pokemon_Model.joblib')

def main():
    pokemon_random_forest = create_model()
    #save_model(pokemon_random_forest)
    

if __name__ == '__main__':
     main()

