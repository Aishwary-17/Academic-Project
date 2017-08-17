'''
Created on 23-Mar-2017

@author: aishwary
'''
import csv
import pandas as pd
import numpy
#-------------------------importing test data and removing unnecessary columns-------------------------------#
df=pd.read_csv('testdata1.csv')
data=pd.DataFrame(df)
data=data.drop('hack_license',axis=1)
data=data.drop('rate_code',axis=1)
data=data.drop('vendor_id',axis=1)
data=data.drop('store_and_fwd_flag',axis=1)
data=data.drop('passenger_count',axis=1)
#-------------------------importing test data and removing unnecessary columns-------------------------------#