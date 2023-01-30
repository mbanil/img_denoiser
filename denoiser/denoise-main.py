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
import h5py
from pathlib import Path
import tifffile

from src import helperfuncs
from src_parallel import classify_parallel
from src_conv import classify_conv
from src import classify
from src import cluster

def main():

    data_dict = {
        "img_folder": "C:/dXs_datalab/Thesis/Implementation/imgs",
        "results_dir" : "C:/dXs_datalab/Thesis/Implementation/results/",
        "img_name" : "18_04_27_Thomas_28618_0017.dm3",
        "reruns" : 15,
        "template_size" : 46,
        "clustering_factor": 2.71,
        "termination_number": 5,
        "analyze": True,
        "min_num_class": 5
    }


    MinNumberInClass=data_dict["min_num_class"]

    folder_path = Path(data_dict["img_folder"])
    img_name = data_dict["img_name"]
    rerun = data_dict["reruns"]
    radius = data_dict["template_size"]//2
    clusteringFactor = data_dict["clustering_factor"]
    analyze = data_dict["analyze"]
    results_dir = data_dict["results_dir"]
    termination_number = data_dict["termination_number"]

    start = time()

    load_start = time()
    imgs = helperfuncs.loadData(folderPath=folder_path, fileName=img_name)
    load_end = time()
    print(f'Time for loading: {load_end - load_start} seconds!')

    startPosList = []
    noOfInititalPatches = np.max(np.asarray(imgs[0].shape)//100)
    if noOfInititalPatches<2:
        noOfInititalPatches = 2
    for i in range(noOfInititalPatches):
        lower_limit = int((i/noOfInititalPatches)*(imgs[0].shape[0]-2*radius))
        upper_limit = int(((i+1)/noOfInititalPatches)*(imgs[0].shape[0]-2*radius))
        rand_X = randint(lower_limit,upper_limit)
        startPosList.append([rand_X, randint(0,imgs[0].shape[1]-2*radius)])

    # startPosList= [[84-radius,404-radius],[97-radius,404-radius],[88-radius,404-radius]]
    print(startPosList)
    MaxNumberInClass=100*int(np.ceil(np.sqrt(len(imgs))))

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
                if(len(templatesCount)>termination_number):
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
                if(len(templatesCount)>termination_number):
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
            if(len(templatesCount)>termination_number):
                if(templatesCount[-1]==len(templates)):
                    if(templatesCount[-1]==templatesCount[-2]):
                        break
            templatesCount.append(len(templates))
            print(f'Completed iteration')
            rerun-=1
        backplot, min, max, templateMatchingResults = classify_conv.backplotImg(radius, imgs, templates)
        
    classify_end = time()
    print(f'Time for generating extra templates and classifying {(rerun_-rerun)} times: {classify_end - classsify_start} seconds!')
    
    sort_start = time()
    picDic = cluster.sortTemplates(imgs, templateMatchingResults, radius, templates)
    sort_end = time()
    print(f'Time for sort : {sort_end - sort_start} seconds!')

    if analyze:

        for i in range(len(imgs)):
            plt.figure(figsize=(2*15, 2*7)) 
            ax1=plt.subplot(1,2,1)                    
            ax1.imshow(imgs[i],cmap=plt.cm.gray)
            ax1.set_title('original image')
            ax1.axis('off')
            ax2=plt.subplot(1,2,2)                    
            ax2.imshow(backplot[i],cmap=plt.cm.gray)
            ax2.set_title('backplot')
            ax2.axis('off')
            #plt.figure(figsize=(15, 12))  
            #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
            #plt.colorbar()
            # plt.show()


        templateClassesMap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
        i=1
        for pic in picDic:
            for p in pic:
                templateClassesMap[p["xIndex"]:p["xIndex"]+10,p["yIndex"]:p["yIndex"]+10]=i
            i+=1
        fig = px.imshow(templateClassesMap, color_continuous_scale=px.colors.qualitative.Alphabet)
        plotly.offline.plot(fig, filename='./charts/'+img_name+'-templateClasses.html')

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
        plotly.offline.plot(fig, filename='./charts/'+img_name+'-noOfPatcherPerPixel.html')

    backplot_start = time()
    backplotFinal, min, max, overlayVariance = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)    
    backplotFinal = helperfuncs.adjustEdges(backplotFinal, imgs)
    backplot_end = time()
    print(f'Time for backplotting-2 : {backplot_end - backplot_start} seconds!')

    for i in range(len(imgs)):
        plt.figure(figsize=(2*15, 2*7)) 
        ax1=plt.subplot(1,2,1)                    
        ax1.imshow(imgs[i],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax1.set_title('original image')
        ax1.axis('off')
        ax2=plt.subplot(1,2,2)                    
        ax2.imshow(backplotFinal[i],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax2.set_title('Denoised Image')
        ax2.axis('off')
        #plt.figure(figsize=(15, 12))  
        #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
            #plt.colorbar()

    # plt.show()

    fig = px.imshow(np.sqrt(overlayVariance[0][radius:-radius,radius:-radius]))
    plotly.offline.plot(fig, filename='./charts/'+img_name+'-overlayVariance.html')

    if analyze:
        fig = px.imshow(np.sqrt(overlayVariance[0][radius:-radius,radius:-radius]))
        plotly.offline.plot(fig, filename='./charts/'+img_name+'-overlayVariance.html')

        fig = px.imshow(backplotFinal[0][radius:-radius,radius:-radius])
        plotly.offline.plot(fig, filename='./charts/'+img_name+'-backplotFinal.html')

        fig = px.imshow((backplotFinal[0][radius:-radius,radius:-radius] - imgs[0][radius:-radius,radius:-radius])**2)
        plotly.offline.plot(fig, filename='./charts/'+img_name+'-diff.html')

        tifffile.imsave(results_dir + img_name + '-denoised.tiff', backplotFinal[0][radius:-radius,radius:-radius])
        tifffile.imsave(results_dir + img_name + '.tiff', imgs[0][radius:-radius,radius:-radius])

        for i in range(len(imgs)):
            plt.figure(figsize=(2*15, 2*7)) 
            ax1=plt.subplot(1,2,1)                    
            ax1.imshow(imgs[i],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
            ax1.set_title('original image')
            ax1.axis('off')
            ax2=plt.subplot(1,2,2)                    
            ax2.imshow(backplotFinal[i],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
            ax2.set_title('Denoised Image')
            ax2.axis('off')
            #plt.figure(figsize=(15, 12))  
            #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
            #plt.colorbar()

        # plt.show()
         
        # plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/results/parallel-stack-'+img_name+'-denoised.png')    

        for i in range(len(imgs)):
            plt.figure(figsize=(20,20))
            img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(imgs[i]))))
            ax1=plt.subplot(1,2,1)
            ax1.imshow(img,cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of original image')
            img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(backplotFinal[i]))))
            ax1=plt.subplot(1,2,2)
            ax1.imshow(img,cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of denoised image in')

        # plt.show()

    with h5py.File('results/'+img_name+'.h5', 'w') as hf:
        hf.create_dataset(img_name,  data=backplotFinal)

    end = time()
    print(f'Total time: {end - start} seconds!')

    return backplotFinal



if __name__ == '__main__':
    freeze_support()
    main()