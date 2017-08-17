'''
Created on 24-Mar-2017

@author: aishwary
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as mp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import pdist, squareform
from numba.tests.npyufunc.test_ufunc import dtype
from scipy.spatial.ckdtree import scipy
import matplotlib.cm as cm
daf=pd.read_csv('processeddata.csv')#--------------------------read the processed data-----------------------------------------#
#---------Apply K-means clustering to the dataset with varying number of clusters----------------------------------------------#
for ncluster in range(10,3,-1):
    
    datai=pd.DataFrame(daf)
    datalist=list(datai)
    datai=datai.drop('Unnamed: 0.1',axis=1)#--------------------------------dropping unnecessary columns-----------------------#
    datai=datai.drop('Unnamed: 0',axis=1)#--------------------------------dropping unnecessary columns-----------------------#
    print list(datai)
    dataarray=datai.drop('medallion',axis=1).as_matrix()#-----------------convert dataset to numpy array data structure for fast and easy calculations-------------------#
    #print dataarray
    kmeans_model =KMeans(n_clusters=ncluster, random_state=1)
    scalar=StandardScaler()#-----------------------data reduction for fast processing--------------------------------------------#
    goodcolumns=scalar.fit_transform(datai._get_numeric_data())#----------------transforming the numerical data-----------------------#
    #print goodcolumns
    kmeans_model.fit(goodcolumns)
    labels=kmeans_model.labels_#------------------------ cluster labels for each individual row in data set----------------------------# 
    centers=kmeans_model.cluster_centers_
    #print centers
    datai['cluster']=labels

#--------------------silhoulette score--------------------------------------------------------#
    print list(datai)
    pca=PCA(2)#------------------------------reduce the data in two dimensions to plot the graph--------------------------------#
    plotc=pca.fit_transform(goodcolumns)
    plotv=pca.fit_transform(centers)
    arr=np.array(centers)
    print 'Arrayofcenterpoitns------------------',arr
#------------------------calculating silhoulette score for each row in the data set----------------------------------------------#
    centermap =[]
    distance_array=scipy.spatial.distance.cdist(arr[:][:],arr[:][:],metric='euclidean')#----distance between centers of clusters---#
    for i in range(len(distance_array)):
            min=100000000.0
            minindex=-1
            for j in range(len(distance_array[i])):
                if distance_array[i][j]!=0:
                        if distance_array[i][j]<min:
                            min=distance_array[i][j]
                            minindex=j
            centermap.append(minindex)
    centermaparray= np.array(centermap)#-----mapping clusters with their nearest clusters
    print centermaparray
    #point=np.array(goodcolumns.ix[:,:])
    #print point
    outlier=np.array(datai.drop('cluster',axis=1)._get_numeric_data())
    silhoulletescorelist=[]#-------------------list to store the silhouellete score for each row--------------------------------------------------------------#
    #print outlier
    for i in range(len(datai)):
            print i
            point=[]
            for j in range(2):
                  point.append(outlier[i])
            pointarray=np.array(point)
            #print point
            key=datai.ix[i,'cluster']
            value=centermaparray[key]
            #print key
            cluster_dataframe=datai.groupby(datai['cluster']).get_group(key)#------------get all the rows with same cluster---------------#
            nearestcluster_dataframe=datai.groupby(datai['cluster']).get_group(value)#-------------------get all the rows of nearest cluster----------------#
            nearestcluster_array=np.array(nearestcluster_dataframe.drop('cluster',axis=1)._get_numeric_data())
            cluster_array=np.array(cluster_dataframe.drop('cluster',axis=1)._get_numeric_data())
            distancearray=scipy.spatial.distance.cdist(pointarray[:][:],cluster_array[:][:],metric='euclidean')#---distance between the point and all the points in the same cluster----#
            nearestclusterdistancearray=scipy.spatial.distance.cdist(pointarray[:][:],nearestcluster_array[:][:],metric='euclidean')
            a= np.mean(distancearray[0])
            b=np.mean(nearestclusterdistancearray[0])
            silhoulletescorelist.append((b-a)/max(b,a))
    silhouette_avg=np.mean(np.array(silhoulletescorelist))
    datai['score']=silhoulletescorelist
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(datai) + (ncluster) * 10])
    y_lower = 10
    for i in range(ncluster):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_values = \
                np.array(datai.groupby(datai['cluster']).get_group(i)['score'])

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral((float(i) / ncluster))
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % ncluster),
                 fontsize=14, fontweight='bold')

    plt.show()    
