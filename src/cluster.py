from copy import deepcopy
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage  
from scipy.cluster.hierarchy import fcluster

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
                "template": img[idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)],
                "index": i
            }
            if(picDic[jt]==None):
                picDic[jt] = [temp]
            else:
                picDic[jt].append(temp)
            picdicindex+=1
        
    print("Used "+str(picdicindex)+"subimages")    
    if errorshappend>0:
        print(str(errorshappend)+"subimages were not included")    
  
    return picDic
        



def cluster(radius, templates, picDic):
    # generating a smooth transistion map:
    backplotwindow=np.zeros((2*radius,2*radius))
    x = np.linspace(0, 1, backplotwindow.shape[0])
    y = np.linspace(0, 1,  backplotwindow.shape[1])
    xv, yv = np.meshgrid(x, y, sparse=True)
    backplotwindow=np.exp(-((4*np.maximum(0,(xv-0.5))**2-0.1)+(4*np.maximum(0,(yv-0.5))**2-0.1)))

    centroidDic=[]
    for jt in range(len(templates)):
        n=0
        # here a n error could appear, its better to check the max size

        templateCount = len(picDic[jt])
        templateShape_reshaped = int(picDic[jt][0]["template"].shape[0]*picDic[jt][0]["template"].shape[1])
        templateShape = picDic[jt][0]["template"].shape

        temp=np.zeros((templateCount,templateShape_reshaped))
        for j in range(templateCount):
            temp[n,:]=(picDic[jt][j]["template"]*backplotwindow).flatten()
            n+=1
        linked = linkage(temp,method='ward')


        # mycriterion='maxclust'
        improvSNR=2.71
        numberofClusters=np.int(np.ceil((templateCount /(improvSNR**2))))

        clusters_=fcluster(linked,  numberofClusters, criterion='maxclust')

        centroid=[np.zeros(templateShape)]*numberofClusters
        centroidCounter=[0]*numberofClusters

        for j in range(len(clusters_)):
            centroid[clusters_[j]-1] += picDic[jt][j]["template"]
            centroidCounter[clusters_[j]-1]+=1

        
        for jc in range(numberofClusters):
            centroid[jc]/=max(centroidCounter[jc],1)

        
        subCentroid = []
        for jc in range(numberofClusters):
            subCentroid.append({
                "centroid": centroid[jc],
                "count": centroidCounter[jc]
            })
        
        
        centroidDic.append(subCentroid)
            

    return centroidDic



















