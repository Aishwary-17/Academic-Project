'''
Created on 24-Mar-2017

@author: aishwary
'''
#-------------removing rows with null values------------------------------------------------------------#
import pandas as pd
daf=pd.read_csv('semiprocesseddata.csv')
semidata=pd.DataFrame(daf)
print list(semidata)
for i in range(10):
 semidata=semidata.drop(i+99984, axis=0)
semidata.to_csv('processeddata.csv')
#-------------removing rows with null values------------------------------------------------------------#