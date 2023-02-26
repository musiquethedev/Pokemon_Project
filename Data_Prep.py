# data preparation file. Not a script
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score


def load_data():
    pokemon_df = pd.read_csv("Natdex_Data.csv")
    return pokemon_df


def create_viability(tier: str):
    '''
    When applied to pokemon_df['tier'], will return a column that represents
    whether the pokemon is good in OU or not.
    
    Parameters:
        tier (str): must be one of the recognized tiers. Exception will be
        thrown otherwise
        
    Returns:
        bool: true for viable, false for not
    '''
    viable_list = ['OU', 'Uber', 'AG']
    # extra entries are added to unviable_list to future proof against tiers
    # that currently have no pokemon
    unviable_list = ['UUBL', 'UU', 'RUBL', 'RU', 'NFE', 'LC', 'LC Uber',
                     '(OU)', '(UU)', '(RU)']

    if (tier in viable_list):
        return True
    elif (tier in unviable_list):
        return False
    else:
        raise Exception("Unidentified Tier")
        


def data_pipeline(pokemon_df, drop_prevos=True, kind='minmax', valid_size=0.1,
                  full_data=False, get_scaler=False):
    '''
    Preps the data from Natdex_Data.csv. It does this by 
    1. loading the data from the csv file
    2. creating the target feature 'isViable'
    3. ordinally encoding the boolean columns, including the target feature
    4. creating dummy features for the 18 types in 'type1'
    5. manually setting the values in these columns for 'type2'
    6. manually removing the outliers of Shedinja, Slaking, and Regigigas
    7. scaling the numerical statistical data according to the parameter 'kind'
    
    
    
    I chose to encode type2 this way because type1 and type2 are inherently
    similar, meaning they should be encoded in the same values
    
    Parameters
    ----------
    pokemon_df: pd.DataFrame
        the dataframe containing the raw data from Natdex_Data.csv
    drop_prevos: boolean
        if True, drops all the pre-evolutions from the dataset
    kind: string: 'minmax' or 'normal', default='minmax'
        determine how to scale the numerical data: 'minmax' scales the data
        in a minmax fashion using sklearn.preprocessing.MinMaxScaler, while
        'normal' us sklearn.preprocessing.StandardScaler to scale and normalize
        the data
    valid_size: float
        determine what proportion of the data set will be kept aside for
        validation
    full_data: 
    
    Returns
    -------
    X, y, X_valid, y_valid: tuple
        X: pd.DataFrame
            the features of the pokemon dataset, scaled, encoded, and with
            outliers removed
        y: pd.Series
            the target feature of the pokemon dataset
        X_valid: pd.DataFrame
            a portion of the dataset that is held aside for final validation
        y_valid: pd.Series
            the labels for X_valid

    '''
    
    
    #add the feature-engineered target tier
    pokemon_df['isViable'] = pokemon_df['tier'].apply(create_viability)
    
    #drop the pre-evolutions according to the parameter
    if (drop_prevos):
        pokemon_df = pokemon_df[pokemon_df['isFinal']]
    
    #ordinally-encode alternate, isLegend, isFinal, and isViable
    pokemon_df['alternate'] = pd.factorize(pokemon_df['alternate'])[0]
    pokemon_df['isLegend'] = pd.factorize(pokemon_df['isLegend'])[0]
    pokemon_df['isFinal'] = pd.factorize(pokemon_df['isFinal'])[0]
    pokemon_df['isViable'] = pd.factorize(pokemon_df['isViable'])[0]
    
    #manually encode type1 and type2 (onehot encoding with the intersection)
    type_list = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire',
                 'flying', 'ghost', 'grass', 'ground', 'ice', 'normal', 'poison',
                 'psychic', 'rock', 'steel', 'water']
    temp = pd.get_dummies(pokemon_df['type1'])
    pokemon_df = pd.concat([pokemon_df, temp], axis='columns')
    for thistype in type_list:
        pokemon_df.loc[pokemon_df['type2'] == thistype, thistype] = 1
    
    #drop unneeded columns
    DROP_COLS = ['name', "ability1", 'ability2', 'hiddenability', 'isFinal', 'tier', 'type1', 'type2']
    pokemon_df = pokemon_df.drop(labels=DROP_COLS, axis=1)
    
    #Remove outliers
    #dropping the outliers of shedinja, regigas, slaking
    pokemon_df = pokemon_df.drop(291, axis=0)
    pokemon_df = pokemon_df.drop(485, axis=0)
    pokemon_df = pokemon_df.drop(288, axis=0)
    
    #Scale the data according to the parameter
    if (kind == 'minmax'):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(copy=True)
    elif (kind == 'normal'):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(copy=True)
    
    scale_columns = ['hp', 'atk', 'physdef', 'spatk', 'spdef', 'speed', 'bst']
    pokemon_df[scale_columns] = scaler.fit_transform(pokemon_df[scale_columns])
    
    
    
    #split into features and target
    
    X = pokemon_df.drop('isViable', axis=1)
    y = pokemon_df['isViable']
    
    #split into validation and usage data
    if (get_scaler):
        return X, y, scaler
    if (full_data):
        
        return X, y
    from sklearn.model_selection import train_test_split
    X, X_valid, y, y_valid = train_test_split(X, y, test_size=valid_size, 
                                              stratify=y)
    
    return X, X_valid, y, y_valid

def test_model(model, X, y, print_info=False):
    '''
    Using the inputted model: this function
    1. uses StratifiedKFold cross validation with 3 splits to ensure accuracy
    2. calculates the f1 score of each split
    3. prints the model type as well as the average f1 score for all splits
    4. returns the list of the f1 scores for each fold

    Parameters
    ----------
    model : any sklearn model
        The model that is being tested. As long as it uses the .fit() and
        .predict() methods, it will work fine with this function
    X : pd.DataFrame
        The features of the dataset. Must be numerical and ideally scaled to
        work with all model types
    y : pd.Series
        The labels for X
    print_info : bool
        if True, prints model information to system console. default is false

    Returns
    -------
    list
        a python list containing the f1 score of each fold

    '''
    #precision and recall used because of imbalanced data
    f1scores = []
    acc_scores = []
    precisionscores = []
    recallscores = []
    
    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1score = f1_score(y_test, y_pred)
        f1scores.append(f1score)
        
        acc_score = accuracy_score(y_test, y_pred)
        acc_scores.append(acc_score)
        
        prec_score = precision_score(y_test, y_pred)
        precisionscores.append(prec_score)
        
        rec_score = recall_score(y_test, y_pred)
        recallscores.append(rec_score)
    if (print_info):
        print(type(model))
        print("Mean f1 score:", np.mean(f1scores))
        print("Mean accuracy score:", np.mean(acc_scores))
        print("Mean Precision score:", np.mean(precisionscores))
        print("Mean recall score:", np.mean(recallscores))
    return {'f1':np.mean(f1scores), 'acc':np.mean(acc_scores), 
            'prec':np.mean(precisionscores), 'rec':np.mean(recallscores)}