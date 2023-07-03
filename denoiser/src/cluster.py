from copy import deepcopy
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage  
from scipy.cluster.hierarchy import fcluster

import matplotlib.pyplot as plt

def sortTemplates(imgs, templateMatchingResults, radius, templates):
    picDic = [None]*len(templates)
    picdicindex=0    
    errorshappend=0

    for i in range(len(imgs)):
        idxd=templateMatchingResults["sortedIndices"][i]
        maxresult=templateMatchingResults["maxresults"][i]
        maxresultindex=templateMatchingResults["maxresultindices"][i]
        img=imgs[i]            
        for j in range(len(idxd[0])):
            jt=int(maxresultindex[idxd[0][j],idxd[1][j]])
            temp = {
                "xIndex": idxd[0][j],
                "yIndex": idxd[1][j],
                "template": deepcopy(img[idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)]),
                "imgIndex": i
            }
            if(picDic[jt]==None):
                picDic[jt] = [temp]
            else:
                picDic[jt].append(deepcopy(temp))
            picdicindex+=1
        
    i = 0
    for p in range(len(picDic)):
        if picDic[i] == None:
            del picDic[i]
            i -= 1
        i += 1

        
    print("Used "+str(picdicindex)+"subimages")    
    if errorshappend>0:
        print(str(errorshappend)+"subimages were not included")    
  
    return deepcopy(picDic)
        



def cluster(radius, templates, picDic, improvSNR):
    # generating a smooth transistion map:
    backplotwindow=np.zeros((2*radius,2*radius))
    x = np.linspace(0, 1, backplotwindow.shape[0])
    y = np.linspace(0, 1,  backplotwindow.shape[1])
    xv, yv = np.meshgrid(x, y, sparse=True)
    backplotwindow=np.exp(-((4*np.maximum(0,(xv-0.5))**2-0.1)+(4*np.maximum(0,(yv-0.5))**2-0.1)))
    if(len(templates[0].shape)>2):
        backplotwindow = np.repeat(backplotwindow[..., None],templates[0].shape[2],axis=2)

    centroidDic=[]
    for jt in range(len(templates)):
        n=0
        # here a n error could appear, its better to check the max size

        templateCount = len(picDic[jt])
        templateShape_reshaped = int(picDic[jt][0]["template"].shape[0]*picDic[jt][0]["template"].shape[1])
        if(len(templates[0].shape)>2):
            templateShape_reshaped = int(picDic[jt][0]["template"].shape[0]*picDic[jt][0]["template"].shape[1]*picDic[jt][0]["template"].shape[2])
        templateShape = picDic[jt][0]["template"].shape

        temp=np.zeros((templateCount,templateShape_reshaped))
        for j in range(templateCount):
            temp[n,:]=deepcopy((picDic[jt][j]["template"]*backplotwindow).flatten())
            n+=1
        linked = linkage(temp,method='ward')
        numberofClusters=np.int(np.ceil((templateCount /(improvSNR**2))))
        clusters_=fcluster(linked,  numberofClusters, criterion='maxclust')

        centroid=[None]*numberofClusters
        if(len(templates[0].shape)>2):
            variance= [np.zeros((2*radius,2*radius,templates[0].shape[2])) for i in range(numberofClusters)] 
        else:
            variance= [np.zeros((2*radius,2*radius)) for i in range(numberofClusters)] 
        centroidCounter=[0]*numberofClusters
        clusterList = [None]*numberofClusters

        for j in range(len(clusters_)):
            centroidCounter[clusters_[j]-1]+=1
            if(clusterList[clusters_[j]-1]==None):
                centroid[clusters_[j]-1] = deepcopy(picDic[jt][j]["template"])
                clusterList[clusters_[j]-1]=[j]
            else:
                centroid[clusters_[j]-1] += deepcopy(picDic[jt][j]["template"])
                clusterList[clusters_[j]-1].append(j)

        for jc in range(numberofClusters):
            centroid[jc]/=max(centroidCounter[jc],1)

        for j in range(len(clusters_)):
            variance[clusters_[j]-1] += deepcopy((picDic[jt][j]["template"] - centroid[clusters_[j]-1])**2)


        subCentroid = []
        for jc in range(numberofClusters):
            subCentroid.append({
                "centroid": deepcopy(centroid[jc]),
                "id": deepcopy(clusterList[jc]),
                "variance_M2": deepcopy(variance[jc]) 
            })
        
        
        centroidDic.append(deepcopy(subCentroid))
            

    return centroidDic


def backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults):

    # generating a smooth transistion map:
    backplotwindow=np.zeros((2*radius,2*radius))
    x = np.linspace(0, 1, backplotwindow.shape[0])
    y = np.linspace(0, 1,  backplotwindow.shape[1])
    xv, yv = np.meshgrid(x, y, sparse=True)
    backplotwindow=np.exp(-((4*np.maximum(0,(xv-0.5))**2-0.1)+(4*np.maximum(0,(yv-0.5))**2-0.1)))   

    # if it is a 3D image
    if(len(imgs.shape)>3):
        backplotwindow = np.repeat(backplotwindow[..., None],imgs.shape[3],axis=3)


    count=0
    pltminradius=3


    overlay=[]
    overlayCount=[]
    overlayclass=[]
    overlayM2=[]
    overlayVariance=[]

    for i in range(len(imgs)):
        img = imgs[i]
        overlay.append(np.zeros(img.shape))
        overlayCount.append(np.zeros(img.shape))
        overlayclass.append(np.zeros(img.shape))
        overlayM2.append(np.zeros(img.shape))
        overlayVariance.append(np.zeros(img.shape))
    
    for jt in range(len(centroidDic)):
        for jc in range(len(centroidDic[jt])):    
            for j in centroidDic[jt][jc]["id"]: 
                myindex=picDic[jt][j]["imgIndex"]  
                maxresultindex=templateMatchingResults["maxresultindices"][myindex]          
                x=picDic[jt][j]["xIndex"]  
                y=picDic[jt][j]["yIndex"]  

                n = backplotwindow+overlayCount[myindex][x:(x+2*radius),(y):(y+2*radius)]
                old_avg = (overlay[i][x:(x+2*radius),(y):(y+2*radius)])/ (overlayCount[i][x:(x+2*radius),(y):(y+2*radius)] + (np.double(overlayCount[i][x:(x+2*radius),y:(y+2*radius)]==0)))
                delta = centroidDic[jt][jc]["centroid"] - old_avg
                overlayM2[i][x:(x+2*radius),(y):(y+2*radius)] = overlayM2[i][x:(x+2*radius),(y):(y+2*radius)] + centroidDic[jt][jc]["variance_M2"]  + delta**2*backplotwindow*overlayCount[myindex][x:(x+2*radius),(y):(y+2*radius)]
                overlayVariance[i][x:(x+2*radius),(y):(y+2*radius)] = overlayM2[i][x:(x+2*radius),(y):(y+2*radius)]/(n + (n==0))

                overlay[myindex][x:(x+2*radius),(y):(y+2*radius)]+=centroidDic[jt][jc]["centroid"]*backplotwindow
                overlayCount[myindex][x:(x+2*radius),(y):(y+2*radius)]+=backplotwindow

                count+=1    
    print("Used "+str(count)+"subimages")

    imgBackplots = []
    mymin=[]
    mymax=[]
    for i in range(len(imgs)):
        imgBackplots.append(overlay[i]/ ( overlayCount[i] + (np.double(overlayCount[i]==0))  ) ) 

        try:
            mymin.append(np.min(imgBackplots[i][imgBackplots[i]>np.min(imgBackplots[i][imgBackplots[i]>0])]))
            mymax.append(np.max(imgBackplots[i][imgBackplots[i]>0]))
        except:
            print("Error occured")
        
    return imgBackplots, mymin, mymax, overlayVariance




    