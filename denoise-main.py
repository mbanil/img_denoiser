from operator import le
import matplotlib.pyplot as plt
import numpy as np
from time import time
import multiprocessing as mp
from multiprocessing import freeze_support
from random import randint
import torch
import plotly.express as px
import plotly

from src import helperfuncs
from src_parallel import classify_parallel
from src_conv import classify_conv
from src import classify
from src import cluster


def main():

    # folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-4/NMC111_delith_15000000X_ABF_stack2/' # Maxime/' #sample 2/'
    folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/'
    # imgName = 'NMC111_delith_15000000X_ABF_stack2.dm3'
    imgName = '18_04_27_Thomas_28618_0017.dm3'
    # imgName = 'Stack_zeolite4NaAF__111_001_1-10.tif'
    rerun = 15
    radius = 23
    clusteringFactor = 2.71
    analyze = False

    start = time()

    load_start = time()
    imgs = helperfuncs.loadData(folderPath=folderPath, fileName=imgName)
    load_end = time()
    print(f'Time for loading: {load_end - load_start} seconds!')

    startPosList = []
    noOfInititalPatches = np.max(np.asarray(imgs[0].shape)//100)
    if noOfInititalPatches<2:
        noOfInititalPatches = 2
    for i in range(noOfInititalPatches):
        startPosList.append([randint(0,imgs[0].shape[0]-radius), randint(0,imgs[0].shape[0]-radius)])

    MinNumberInClass=4
    MaxNumberInClass=100*int(np.ceil(np.sqrt(len(imgs))))
    
    if analyze:
        n1_max=1
        n1=0
        n2_max=len(imgs)
        n2=0
        plt.figure(figsize=(20, 20*n2_max/n1_max)) 
        for img in imgs:
            n2+=1    
            vstd=np.std(img)
            vmean=np.mean(img)         
            ax1=plt.subplot(n2_max,n1_max,n1*n2_max+n2)
            ax1.imshow(img,cmap='gray',vmin=np.min(img),vmax=np.max(img))
            ax1.axis('off')
            
        plt.show()

    gen_start = time()
    templates = helperfuncs.generateTemplates(startPosList=startPosList, imgs=imgs, radius=radius)
    gen_end = time()
    print(f'Time for generating basic templates: {gen_end - gen_start} seconds!')

    rerun_ = rerun
    classsify_start = time()
    templatesCount = []
    if len(imgs)>1:
        if torch.cuda.is_available():
            while rerun>0:
                templates = classify_conv.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
                if(len(templatesCount)!=0):
                    if(templatesCount[-1]==len(templates)):
                        if(templatesCount[-1]==templatesCount[-2]):
                            break
                templatesCount.append(len(templates))
                print(f'Completed iteration')
                rerun-=1
            backplot, min, max, templateMatchingResults = classify_conv.backplotImg(radius, imgs, templates)
        else:
            pool = mp.Pool(mp.cpu_count())
            while rerun>0:
                templates = classify_parallel.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass, pool= pool)                
                if(len(templatesCount)!=0):
                    if(templatesCount[-1]==len(templates)):
                        if(templatesCount[-1]==templatesCount[-2]):
                            break
                templatesCount.append(len(templates))
                print(f'Completed iteration')
                rerun-=1
            backplot, min, max, templateMatchingResults = classify_parallel.backplotImg(radius, imgs, templates, pool)
            pool.close()
    else:
        while rerun>0:
            templates = classify_conv.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
            if(len(templatesCount)!=0):
                if(templatesCount[-1]==len(templates)):
                    if(templatesCount[-1]==templatesCount[-2]):
                        break
            templatesCount.append(len(templates))
            print(f'Completed iteration')
            rerun-=1
        backplot, min, max, templateMatchingResults = classify_conv.backplotImg(radius, imgs, templates)
        
    classify_end = time()
    print(f'Time for generating extra templates and classifying {rerun_} times: {classify_end - classsify_start} seconds!')
    
    sort_start = time()
    picDic = cluster.sortTemplates(imgs, templateMatchingResults, radius, templates)
    sort_end = time()
    print(f'Time for sort : {sort_end - sort_start} seconds!')

    if analyze:
        templateClassesMap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
        i=1
        for pic in picDic:
            for p in pic:
                templateClassesMap[p["xIndex"]:p["xIndex"]+10,p["yIndex"]:p["yIndex"]+10]=i
            i+=1
        fig = px.imshow(templateClassesMap, color_continuous_scale=px.colors.qualitative.Alphabet)
        plotly.offline.plot(fig, filename='./charts/'+imgName+'-templateClasses.html')

    cluster_start = time()
    centroidDic = cluster.cluster(radius, templates, picDic, clusteringFactor)
    cluster_end = time()
    print(f'Time for clustering : {cluster_end - cluster_start} seconds!')

    if analyze:
        noOfPatchesPerPixel = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
        for i in range(len(centroidDic)):
            for centroid in centroidDic[i]:
                ids = centroid["id"]
                for id in ids:
                    pos = picDic[i][id]
                    noOfPatchesPerPixel[pos["xIndex"]:pos["xIndex"]+2*radius,pos["yIndex"]:pos["yIndex"]+2*radius]+=len(ids)

        fig = px.imshow(noOfPatchesPerPixel)
        plotly.offline.plot(fig, filename='./charts/'+imgName+'-noOfPatcherPerPixel.html')

    backplot_start = time()
    backplotFinal, min, max, overlayVariance = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)    
    backplot_end = time()
    print(f'Time for backplotting-2 : {backplot_end - backplot_start} seconds!')

    if analyze:
        fig = px.imshow(np.sqrt(overlayVariance[0][radius:-radius,radius:-radius]))
        plotly.offline.plot(fig, filename='./charts/'+imgName+'-overlayVariance.html')

        fig = px.imshow(backplotFinal[0][radius:-radius,radius:-radius])
        plotly.offline.plot(fig, filename='./charts/'+imgName+'-backplotFinal.html')

        fig = px.imshow((backplotFinal[0][radius:-radius,radius:-radius] - imgs[0][radius:-radius,radius:-radius])**2)
        plotly.offline.plot(fig, filename='./charts/'+imgName+'-diff.html')

        for i in range(len(imgs)):
            plt.figure(figsize=(2*15, 2*7)) 
            ax1=plt.subplot(1,2,1)                    
            ax1.imshow(imgs[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=0,vmax=512)
            ax1.set_title('original image')
            ax1.axis('off')
            ax2=plt.subplot(1,2,2)                    
            ax2.imshow(backplotFinal[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=0,vmax=512)
            ax2.set_title('backplot')
            ax2.axis('off')
            #plt.figure(figsize=(15, 12))  
            #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
            #plt.colorbar()
            # plt.show()
         
        plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/results/parallel-stack-'+imgName+'-denoised.png')    

        for i in range(len(imgs)):
            plt.figure(figsize=(20,20))
            img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(imgs[i][radius:-radius,radius:-radius]))))
            ax1=plt.subplot(1,2,1)
            ax1.imshow(img,cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of original image')
            img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(backplotFinal[i][radius:-radius,radius:-radius]))))
            ax1=plt.subplot(1,2,2)
            ax1.imshow(img,cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of denoised image in')
            plt.show()

    end = time()
    print(f'Total time: {end - start} seconds!')

    return backplotFinal

    

if __name__ == '__main__':
    freeze_support()
    main()