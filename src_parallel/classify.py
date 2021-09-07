import numpy as np
from skimage.feature import match_template
from copy import deepcopy

def tempfuncname(radius, imgs, templates, maxNumberInClass, minNumberInClass):

    maxresultindices, maxresults, sortedIndices = classifyTemplates(radius, imgs, templates)

    newTemplates = generateNewTemplates(templates, imgs, sortedIndices, maxresults, maxresultindices, radius, maxNumberInClass, minNumberInClass)
            
    return deepcopy(newTemplates)



def classifyTemplates(radius, imgs, templates):

    minradius = np.int16(radius/2)
    maxresultindices = []
    maxresults = []
    sortedIndices = []

    for img in imgs:
        firstrun=True

        for t in range(len(templates)): 
            result=match_template(img, templates[t])
            resultshape=result.shape
            # changes here required if using multimode
            if firstrun:
                firstrun=False
                maxresultindex=np.zeros(resultshape)
                maxresult=np.zeros(resultshape)    
            maxresultindex[result>maxresult]=t
            maxresult[result>maxresult]=result[result>maxresult]
    
        # Now we rearange these results:
        idx = (-maxresult.flatten()).argsort()
        idxd=np.unravel_index(idx,result.shape)
        goodlist=[]
        for myk in range(len(idx)):
            if  (maxresult[idxd[0][myk],idxd[1][myk]]>0):
                maxresult[max(0,idxd[0][myk]-minradius):min(maxresult.shape[0],idxd[0][myk]+minradius),
                            max(0,idxd[1][myk]-minradius):min(maxresult.shape[1],idxd[1][myk]+minradius)]=0
                goodlist.append(myk)

        idxdnew=np.zeros((2, len(goodlist)), dtype=int)
        n=0
        for myk in goodlist:
            idxdnew[:,n]=[idxd[0][myk],idxd[1][myk]]
            n+=1

        sortedIndices.append(deepcopy(idxdnew))
        maxresultindices.append(deepcopy(maxresultindex))
        maxresults.append(deepcopy(maxresult))
    
    return maxresultindices, maxresults, sortedIndices


def generateNewTemplates(templates, imgs, sortedIndices, maxresults, maxresultindices, radius, maxNumberInClass, minNumberInClass):

    newTemplates = []
    ncount=[]
    templateIDs=[]
    for i in range(len(templates)): #ini new templates
        try:
            newTemplates.append(np.zeros(templates[i].shape)) 
        except:
            pass
        ncount.append(0)
        templateIDs.append(i)
    
    for i in range(len(imgs)): #generates new templates
        img = imgs[i]
        idxd = sortedIndices[i]
        maxresult=maxresults[i]
        maxresultindex=maxresultindices[i]
        for j in range(len(idxd[0])):
            # try:
            jt=np.int32(maxresultindex[idxd[0][j],idxd[1][j]])                
            newTemplates[templateIDs[jt]]+=deepcopy(img[idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)])
            ncount[templateIDs[jt]]+=1
            if ncount[templateIDs[jt]]>=maxNumberInClass:
                # print(len(newTemplates))
                templateIDs[jt]=max(templateIDs)+1
                ncount.append(0)
                newTemplates.append(np.zeros(templates[0].shape))
            # except:
            #     print('skipt: #j#'+str(j)) 

    i=0
    # print(len(newTemplates))
    for jtnew in range(len(newTemplates)):
        # changes required here for multimode
        if ncount[i]<minNumberInClass:
            del newTemplates[i]
            del ncount[i]
            i-=1
        else:
            newTemplates[i]/=ncount[i]
        i+=1
            
    return deepcopy(newTemplates)


def backplotImg(radius, imgs, templates):
    maxresultindices, maxresults, sortedIndices = classifyTemplates(radius, imgs, templates)

    

    overlay=[]
    overlayCount=[]
    
    pltminradius=3

    for i in range(len(imgs)):
        img = imgs[i]
        overlay.append(np.zeros(img.shape))
        overlayCount.append(np.zeros(img.shape))

    # generating a smooth transistion (gaussian) map:
    backplotwindow=np.zeros((2*radius,2*radius))
    x = np.linspace(0, 1, backplotwindow.shape[0])
    y = np.linspace(0, 1,  backplotwindow.shape[1])
    xv, yv = np.meshgrid(x, y, sparse=True)
    backplotwindow=np.exp(-((4*np.maximum(0,(xv-0.5))**2-0.1)+(4*np.maximum(0,(yv-0.5))**2-0.1)))
    
    # the back plotting
    n=0
    for i in range(len(sortedIndices)):
        idxd = sortedIndices[i]
        for j in range(idxd.shape[1]):
            # try:
                #print(j)
            templateIdx = int(maxresultindices[i][idxd[0][j],idxd[1][j]]) 
            overlay[i][idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)]+= deepcopy(templates[templateIdx]*backplotwindow)
            overlayCount[i][idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)]+=backplotwindow
            n+=1    
            # except:
            #     pass
    print("Used "+str(n)+"subimages")

    imgBackplots = []
    mymin=[]
    mymax=[]
    for i in range(len(imgs)):
        imgBackplots.append(overlay[i]/ ( overlayCount[i] + (np.double(overlayCount[i]==0))  ) ) 
        mymin.append(np.min(imgBackplots[i][imgBackplots[i]>np.min(imgBackplots[i][imgBackplots[i]>0])]))
        mymax.append(np.max(imgBackplots[i][imgBackplots[i]>0]))

    templateMatchingResults = {
        "maxresultindices": deepcopy(maxresultindices),
        "maxresults": deepcopy(maxresults),
        "sortedIndices": deepcopy(sortedIndices)
    }

    return imgBackplots, mymin, mymax, templateMatchingResults


