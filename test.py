# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:43:08 2015

@author: junwan
"""


import logging
import numpy as np
import pandas as pd
from scipy import sparse
import cPickle as pickle
from itertools import combinations

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="history.log", filemode='a', level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')

logger = logging.getLogger(__name__)

path = 'C:/Users/junwan/Desktop/Projects/K/RossmannStoreSales/data/'

def load_data(filename, DtVar = None):
    logging.debug("loading data from %s", filename)
    data = pd.read_csv(path + filename, parse_dates = DtVar, delimiter = ",").rename(columns = str.lower)
    return data
    
def save_dataset(filename, X, X_test, features=None, features_test=None):
    """Save the training and test sets augmented with the given features."""
    if features is not None:
        assert features.shape[1] == features_test.shape[1], "features mismatch"
        if sparse.issparse(X):
            features = sparse.lil_matrix(features)
            features_test = sparse.lil_matrix(features_test)
            X = sparse.hstack((X, features), 'csr')
            X_test = sparse.hstack((X_test, features_test), 'csr')
        else:
            X = np.hstack((X, features))
            X_test = np. hstack((X_test, features_test))

    logger.info("> saving %s to disk", filename)
    with open(path + "%s.pkl" % filename, 'wb') as f:
        if filename[-2:] =='_L':
            np.save(f, X)
            np.save(f, X_test)
        else:
            pickle.dump((X,X_test), f, pickle.HIGHEST_PROTOCOL)
        f.close()

def j1way_cnt(dsn, var1):
     df = dsn[[var1]]
     df['cnt'] = 1
     return df.groupby([var1]).transform(np.sum)
     
def j2way_cnt(dsn, var1, var2):
     df = dsn[[var1, var2]]
     df['cnt'] = 1
     return df.groupby([var1, var2]).transform(np.sum)

def j3way_cnt(dsn, var1, var2, var3):
     df = dsn[[var1, var2, var3]]
     df['cnt'] = 1
     return df.groupby([var1, var2, var3]).transform(np.sum)


train = load_data('train.csv', [2])
test = load_data('test.csv', [3])
store = load_data('store.csv')

train = pd.merge(train, store, on='store', how='left')
test = pd.merge(test, store, on='store', how='left')

train = train.loc[train.sales > 0]
train['year'] = pd.DatetimeIndex(train['date']).year
train['month'] = pd.DatetimeIndex(train['date']).month
train = train[train.month < 10]
train = train[np.logical_and(train.stateholiday != 'b', train.stateholiday != 'c')]
train['stateholiday'][train.stateholiday != 'a'] = '0' 

test['year'] = pd.DatetimeIndex(test['date']).year
test['month'] = pd.DatetimeIndex(test['date']).month

test = fillCustomers(train, test, ['store', 'dayofweek', 'year', 'promo'])
        
allData = pd.concat([train, test], ignore_index=True)

cat = ['store', 'dayofweek', 'promo', 'stateholiday', 'schoolholiday', 'storetype', 'assortment', 'competitionopensincemonth', 'competitionopensinceyear', 'promo2', 'promo2sinceyear', 'promointerval', 'month']

for var in cat:
    allData['cnt_' + var] =  j1way_cnt(allData, var).cnt

train1 = allData[:train.shape[0]]
test1 = allData[train.shape[0]:]
save_dataset("basic20132015_cnt1", train1, test1)
 
for index in combinations(cat, 2):
    var=list(index)
    allData['cnt_' + var[0] + var[1]] =  j2way_cnt(allData, var[0], var[1]).cnt

for index in combinations(cat, 3):
    var=list(index)
    allData['cnt_' + var[0] + var[1] + var[2]] =  j2way_cnt(allData, var[0], var[1], var[2]).cnt














