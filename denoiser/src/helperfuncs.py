from copy import deepcopy
from skimage import io
import hyperspy.api as hs
import numpy as np
from skimage.feature import match_template
import os


def loadData(folderPath, fileName):
    imgs = []

    img_path = os.path.join(folderPath, fileName)

    if fileName.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
        if len(io.imread(img_path).shape)>2:
            b= np.float64(io.imread(img_path))
            for j in range(b.shape[0]):
                imgs.append(b[j,:,:])
        else:   
            imgs.append(np.float64(io.imread(img_path)))
    elif fileName.lower().endswith(('.dm3','.emd')):
        dm3Data=hs.load(img_path)
        if fileName.lower().endswith(('.emd')):
            data = []
            for channel in dm3Data:
                data.append(np.float64(channel.data))
            data = np.stack(data, axis=2)
            imgs.append(data)
        else:
            imgs.append(np.float64(dm3Data.data))
            if len(imgs[0].shape)>2:
                imgs = list(imgs[0])
    else:
        print('ERROR: filetyp not supported! Please contact me.')


    return imgs


def generateTemplates(intial_patch_locations, image, radius):
    templates=[]

    # templates are only taken from the first image of the imges list (since the images are similar)
    for start_pos in intial_patch_locations:
        templates.append(deepcopy(image[0,start_pos[0]:start_pos[0]+np.int(2*radius),
                                         start_pos[1]:start_pos[1]+np.int(2*radius)]))

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


def adjustEdges(backPlots, imgs):
    for k in range(len(imgs)):
        for i in range(imgs[k].shape[0]):
            for j in range(imgs[k].shape[1]):
                if(backPlots[k][i,j]==0):
                    backPlots[k][i,j] = imgs[k][i,j]

    return deepcopy(backPlots)


# create backplot window