
# import hyperspy.api as hs
# import numpy as np 
# import scipy.special 
# import scipy.signal
# import scipy.interpolate
# import scipy.linalg
# import scipy.fftpack
# import scipy.misc
# from scipy.spatial import distance
# import imageio
# from skimage.feature import match_template
# from skimage import io
# from functools import reduce
# from copy import deepcopy
# import matplotlib.pyplot as plt
# %matplotlib qt

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import centroid

from src import helperfuncs
from src import classify
from src import cluster

folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/'
imgName = '18_04_27_Thomas_28618_0016.dm3'

radius=23
NumMainclasses=4
MinNumberInClass=4
MaxNumberInClass=100
myrun=0
rerun = 15

imgs = helperfuncs.loadData(folderPath=folderPath, fileName=imgName)


# n1_max=1
# n1=0
# n2_max=len(imgs)
# n2=0
# plt.figure(figsize=(20, 20*n2_max/n1_max)) 
# for img in imgs:
#     n2+=1    
#     vstd=np.std(img)
#     vmean=np.mean(img)         
#     ax1=plt.subplot(n2_max,n1_max,n1*n2_max+n2)
#     ax1.imshow(img,cmap='gray',vmin=max(vmean-9*vstd,np.min(img)),vmax=min(vmean+9*vstd,np.max(img)))
#     ax1.axis('off')

# plt.show()

startPosList= [[84-radius,404-radius],[97-radius,404-radius],[88-radius,404-radius]]
templates = helperfuncs.generateTemplates(startPosList=startPosList, imgs=imgs, radius=radius)

templates = helperfuncs.findDissimilarTemplates(templates = templates, imgs = imgs, radius = radius, minTemplateClasses = NumMainclasses)

# n1_max= 1
# n1=0
# n2_max=len(templates)
# n2=0
# plt.figure(figsize=(20, 20*n2_max/n1_max)) 
# for template in templates:
#     n2+=1
#     plt.subplot(n2_max, n1_max, n1*n2_max+n2)
#     plt.imshow(template, cmap='gray')
#     plt.axis('off')

# plt.show()

while rerun>0:
    templates = classify.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
    rerun-=1

print(len(templates))



# n1_max=len(list(FileDic.keys()))
# n1=-1
# n2_max=max([len(list(newtemplate[Mode].keys())) for Mode in FileDic.keys()])
# plt.figure(figsize=(20, 20*n1_max/n2_max)) 
# for Mode in FileDic.keys():
#     n1+=1
#     n2=0
#     for jtnew in newtemplate[Mode].keys():
#         try:
#             n2+=1
#             plt.subplot(n1_max, n2_max, n1*n2_max+n2)
#             plt.imshow(newtemplate[Mode][jtnew])
#             plt.title('Nr: '+str(jtnew))
#             plt.axis('off')
#         except:
#             pass


backplot, min, max, templateMatchingResults = classify.backplotImg(radius, imgs, templates)



for i in range(len(backplot)):
    max = np.max(backplot[i])
    min = np.min(backplot[i])
    plt.figure(figsize=(15, 7)) 
    ax1=plt.subplot(1,2,1)                    
    ax1.imshow(backplot[i],cmap=plt.cm.gray,vmin=min,vmax=max)
    ax1.set_title('backplot')
    ax1.axis('off')
    ax2=plt.subplot(1,2,2)                    
    ax2.imshow(imgs[i],cmap=plt.cm.gray,vmin=min,vmax=max)
    ax2.set_title('original image')
    ax2.axis('off')
    # plt.figure(figsize=(15, 12))  
    # plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
    # plt.colorbar()
    plt.show()

picDic = cluster.sortTemplates(imgs, templateMatchingResults, radius, templates)

centroidDic = cluster.cluster(radius, templates, picDic)

backplotFinal, min, max = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)

for i in range(len(imgs)):
    plt.figure(figsize=(2*15, 2*7)) 
    ax1=plt.subplot(1,2,1)                    
    ax1.imshow(backplotFinal[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
    ax1.set_title('backplot')
    ax1.axis('off')
    ax2=plt.subplot(1,2,2)                    
    ax2.imshow(imgs[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
    ax2.set_title('original image')
    ax2.axis('off')
    #plt.figure(figsize=(15, 12))  
    #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
    #plt.colorbar()
    plt.show()

print(len(picDic))