import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def get_unique_datetimes(df):
    df = df.reset_index()
    dtm = df.groupby('datetime').agg({'index': list})
    dtm = dtm.reset_index()
    return dtm


def split_dataset(df, tss):
    dtm = get_unique_datetimes(df)
    train_indexes = []
    test_indexes = []
    for train_index, test_index in tss.split(dtm):
        train_indexes.append(np.sum(dtm['index'].iloc[train_index]))
        test_indexes.append(np.sum(dtm['index'].iloc[test_index]))
    return train_indexes, test_indexes


"""
    #Usage example:
    
    tss = TimeSeriesSplit(n_splits=10)
    
    for train_index, test_index in zip(*split_dataset(df, tss)):
    
        print("Train Set:")
        train_set = df.iloc[train_index]
        print(train_set)
        
        print("Test Set:")
        test_set = df.iloc[test_index]
        print(test_set)
                    
        print("-----")
    
    #Pay attention that train_set and test_set might not be sorted as input df! 
"""
