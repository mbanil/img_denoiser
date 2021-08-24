import matplotlib.pyplot as plt
import numpy as np
from time import time
import cProfile
import io
import pstats

from src import helperfuncs
from src import classify
from src import cluster



folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/' #/Maxime/sample 2/'
imgName = '18_04_27_Thomas_28618_0017.dm3'
# startPosList= [[84-radius,404-radius],[97-radius,404-radius],[88-radius,404-radius]]


def denoise(folderPath, imgName, rerun = 15, radius=23):

    start = time()

    startPosList= [[84-radius,404-radius],[97-radius,404-radius],[88-radius,404-radius]]

    load_start = time()
    imgs = helperfuncs.loadData(folderPath=folderPath, fileName=imgName)
    load_end = time()
    print(f'Time for loading: {load_end - load_start} seconds!')

    NumMainclasses=4
    MinNumberInClass=4
    MaxNumberInClass=100


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
    #     ax1.imshow(img,cmap='gray',vmin=np.min(img),vmax=np.max(img))
    #     ax1.axis('off')

    # plt.show()

    startPosList= [[84-radius,404-radius],[97-radius,404-radius],[88-radius,404-radius]]

    gen_start = time()
    templates = helperfuncs.generateTemplates(startPosList=startPosList, imgs=imgs, radius=radius)
    templates = helperfuncs.findDissimilarTemplates(templates = templates, imgs = imgs, radius = radius, minTemplateClasses = NumMainclasses)
    gen_end = time()
    print(f'Time for generating basic templates: {gen_end - gen_start} seconds!')

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
    rerun_ = rerun
    classsify_start = time()
    while rerun>0:
        templates = classify.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
        rerun-=1
    classify_end = time()
    print(f'Time for generating extra templates and classifying {rerun_} times: {classify_end - classsify_start} seconds!')



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

    backplot_start = time()
    backplot, min, max, templateMatchingResults = classify.backplotImg(radius, imgs, templates)
    backplot_end = time()
    print(f'Time for backplotting-1 : {backplot_end - backplot_start} seconds!')
    

    # for i in range(len(backplot)):
    #     max = np.max(backplot[i])
    #     min = np.min(backplot[i])
    #     plt.figure(figsize=(15, 7)) 
    #     ax1=plt.subplot(1,2,1)                    
    #     ax1.imshow(backplot[i],cmap=plt.cm.gray,vmin=min,vmax=max)
    #     ax1.set_title('backplot')
    #     ax1.axis('off')
    #     ax2=plt.subplot(1,2,2)                    
    #     ax2.imshow(imgs[i],cmap=plt.cm.gray,vmin=min,vmax=max)
    #     ax2.set_title('original image')
    #     ax2.axis('off')
    #     # plt.figure(figsize=(15, 12))  
    #     # plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
    #     # plt.colorbar()
    #     plt.show()

    sort_start = time()
    picDic = cluster.sortTemplates(imgs, templateMatchingResults, radius, templates)
    sort_end = time()
    print(f'Time for sort : {sort_end - sort_start} seconds!')

    cluster_start = time()
    centroidDic = cluster.cluster(radius, templates, picDic)
    cluster_end = time()
    print(f'Time for clustering : {cluster_end - cluster_start} seconds!')

    backplot_start = time()
    backplotFinal, min, max = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)    
    backplot_end = time()
    print(f'Time for backplotting-2 : {backplot_end - backplot_start} seconds!')
    

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

    plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/results/'+imgName+'-denoised.png')    

    end = time()
    print(f'Total time: {end - start} seconds!')

# denoise(folderPath, imgName, rerun = 15, radius=23)
# cProfile.run('denoise(folderPath, imgName, rerun = 15, radius=23)')

pr = cProfile.Profile()
pr.enable()
denoise(folderPath, imgName, rerun = 15, radius=23)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats("denoise")
print(s.getvalue())