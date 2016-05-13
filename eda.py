# -*- coding: utf-8 -*-
"""eda.py
Created on Tue Nov 03 15:13:39 2015

@author: Jun Wang
"""
from __future__ import division

import argparse
import logging
import numpy as np
import pandas as pd
from pandas import Series, DataFrame 
from helper.data import load_data
from helper.utils import JExploreDataAnalysis02, JTabulation02

import sys
sys.path.append("..")

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                    filename="eda.log",
                    filemode='a',
                    level=logging.DEBUG,
                    datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                              datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

    
def main():
    logger.info("loading data")
    train = load_data('train.csv', [2])
    logger.info("%d columns and %d rows in train", train.shape[0], train.shape[1])
    test = load_data('test.csv', [3])
    logger.info("%d columns and %d rows in test", test.shape[0], test.shape[1])
    store = load_data('store.csv')
    logger.info("%d columns and %d rows in store", store.shape[0], store.shape[1])    

    #train['stateholiday01'] = '0'   
    #train['stateholiday01'] = [x for x in train.stateholiday if x in ['a','b','c','0']]

    logger.info("merging data")
    train = pd.merge(train, store, on='store', how='left')
    logger.info("%d columns and %d rows in merged train", train.shape[0], train.shape[1])
    test = pd.merge(test, store, on='store', how='left')
    logger.info("%d columns and %d rows in merged test", test.shape[0], test.shape[1])
    
    logger.info("Explore Data Analysis")
    JExploreDataAnalysis02(train, ['date'],'train')
    JExploreDataAnalysis02(test, ['date'], 'test')  

    logger.info("Tabulation Analysis")
    train['customersBin'] = pd.qcut(train.customers,4)
    train['competitiondistanceBin'] = pd.qcut(train.competitiondistance, 4)
    train['promo2sinceweekBin'] = pd.qcut(train.promo2sinceweek, 4)
    varList = ['customersBin', 'dayofweek', 'stateholiday', 'schoolholiday', 'dateYr','dateMth','dateDay','storetype','assortment','competitiondistanceBin','competitionopensincemonth','competitionopensinceyear','promo2sinceweekBin', 'promo2', 'promo2sinceweek','promo2sinceyear', 'promointerval','store']
    trainPositiveSales = train.loc[train.sales > 0]
    JTabulation02(trainPositiveSales, ['sales', 'customers'], varList, 'tabulation')
    trainPositiveSales2015 = trainPositiveSales.loc[trainPositiveSales.dateYr == 2015]
    trainPositiveSales2015a0 = trainPositiveSales2015.loc[trainPositiveSales2015.stateholiday != 'b']    
    JTabulation02(trainPositiveSales2015a0, ['sales', 'customers'], varList, 'tabulation2015a0')

    testStore = DataFrame(test.store.unique(), columns = ['store'])
    trainStore = pd.merge(trainPositiveSales2015a0, testStore, on = 'store', how='inner')
    JTabulation02(trainStore, ['sales', 'customers'], varList, 'tabulation2015a0Store')
    
    
if __name__ == "__main__":
    main()    









