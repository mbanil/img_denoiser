import numpy as np
from skimage.feature import match_template

def tempfuncname(radius, imgs, templates, maxNumberInClass, minNumberInClass):

    maxresultindices, maxresults, sortedIndices = classifyTemplates(radius, imgs, templates)

    newTemplates = generateNewTemplates(templates, imgs, sortedIndices, maxresults, maxresultindices, radius, maxNumberInClass, minNumberInClass)
            
    return newTemplates



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

        sortedIndices.append(idxdnew)
        maxresultindices.append(maxresultindex)
        maxresults.append(maxresult)
    
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
            newTemplates[templateIDs[jt]]+=img[idxd[0][j]:(idxd[0][j]+2*radius),(idxd[1][j]):(idxd[1][j]+2*radius)]
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
            
    return newTemplates