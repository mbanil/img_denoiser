from copy import deepcopy
from skimage import io
import hyperspy.api as hs
import numpy as np
from skimage.feature import match_template


def loadData(folderPath, fileName):
    imgs = [];

    if fileName.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
        if len(io.imread(folderPath + fileName).shape)>2:
            b= np.float64(io.imread(folderPath + fileName))
            for j in range(b.shape[0]):
                imgs.append(b[j,:,:])
        else:   
            imgs.append(np.float64(io.imread(folderPath + fileName)))
    elif fileName.lower().endswith(('.dm3','.emd')):
        dm3Data=hs.load(folderPath + fileName)
        if fileName.lower().endswith(('.emd')):
            data = []
            for channel in dm3Data:
                data.append(np.float64(channel.data))
                break
            data = np.stack(data, axis=2)
            imgs.append(data[:,:,0])
        else:
            imgs.append(np.float64(dm3Data.data))
    else:
        print('ERROR: filetyp not supported! Please contact me.')

    return imgs

def generateTemplates(startPosList, imgs, radius):
    templates=[]

    # templates are only taken from the first image of the imges list (since the images are similar)
    for startPos in startPosList:
        templates.append(deepcopy(imgs[0][startPos[0]:startPos[0]+np.int(2*radius),
                                         startPos[1]:startPos[1]+np.int(2*radius)]))

    return templates
    
def findDissimilarTemplates(templates, imgs, radius, minTemplateClasses):

    minresults=[]

    # best result is at idx 0
    best=0 

    while len(templates)<minTemplateClasses:
        i=0
        for img in imgs:

            if not len(minresults)==len(imgs):
                minresult=[]
                for template in templates:
                    result = match_template(img, template)                
                    resultshape=result.shape
                    if len(minresult)==0:
                        minresult=np.abs(result)
                    else:
                        minresult=np.maximum(minresult,np.abs(result))

                minresults.append(minresult)

            idx = (minresults[i].flatten()).argsort()
            idxd=np.unravel_index(idx,resultshape)

            templates.append(deepcopy(img[idxd[0][best]:idxd[0][best]+np.int(2*radius),
                                            idxd[1][best]:idxd[1][best]+np.int(2*radius)]))
            
            if len(templates) >= minTemplateClasses:
                break

            i+=1
        best+=1
    
    return templates

# create backplot window