'''
Created on 24-Mar-2017

@author: aishwary

Pre-Processing the data
'''
#-------import the reduced data---------------------#
from testmodule import data
print data.get_value(0, 'medallion', False)
rowlength= len(data.index)
print list(data)
print data['trip_distance'].max()#------------------max distance travelled in a route----------------------#
#------------------changing the pickup and dropoff timestamps to minutes for calculation-------------------------------------#
for i in range(rowlength-1):
    pickupdatetime= str(data.ix[i,'pickup_datetime'])
    dropoffdatetime=str(data.ix[i,'dropoff_datetime'])
    ptime=pickupdatetime[11:19]
    dtime=dropoffdatetime[11:19]
    ptime=ptime.split(':')
    dtime=dtime.split(':')
    ptime_in_mins=int(ptime[0])*60 + int(ptime[1])
    dtime_in_mins=int(dtime[0])*60 + int(dtime[1])
    data.set_value(i, 'pickup_datetime', ptime_in_mins)
    data.set_value(i, 'dropoff_datetime', dtime_in_mins)
#------------------changing the pickup and dropoff timestamps to minutes for calculation-------------------------------------#
data.to_csv('semiprocesseddata.csv')
print 'done!'

        
        