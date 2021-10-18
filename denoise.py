import matplotlib.pyplot as plt
import numpy as np
from time import time
import cProfile
import io
import pstats
from numpy.core.fromnumeric import var
import plotly.express as px
import plotly

from src import helperfuncs
from src import classify
from src import cluster


# folderPath ="C:/Users/anilm/Downloads/"
# imgName = "car.png"

folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/' # Maxime/' #sample 2/'
imgName = '18_04_27_Thomas_28618_0017.dm3'
# # folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-4/NMC111_delith_15000000X_ABF_stack2/' # Maxime/' #sample 2/'
# folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-2/'
# # imgName = 'NMC111_delith_15000000X_ABF_stack2.dm3'
# imgName = 'Stack_zeolite4NaAF__111_001_1-10.tif'


def denoise(folderPath, imgName, rerun = 15, radius=23):

    start = time()

    startPosList= [[84-radius,104-radius],[97-radius,204-radius],[88-radius,154-radius]]

    load_start = time()
    imgs = helperfuncs.loadData(folderPath=folderPath, fileName=imgName)
    load_end = time()
    print(f'Time for loading: {load_end - load_start} seconds!')

    NumMainclasses=4
    MinNumberInClass=4
    # MaxNumberInClass=100*int(np.ceil(np.sqrt(len(imgs))))
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

    
    gen_start = time()
    templates = helperfuncs.generateTemplates(startPosList=startPosList, imgs=imgs, radius=radius)
    templates = helperfuncs.findDissimilarTemplates(templates = templates, imgs = imgs, radius = radius, minTemplateClasses = NumMainclasses)
    gen_end = time()
    print(f'Time for generating basic templates: {gen_end - gen_start} seconds!')

    rerun_ = rerun
    classsify_start = time()
    templatesCount = []
    while rerun>0:
        templates = classify.tempfuncname(radius=radius, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
        if(len(templatesCount)!=0):
            if(templatesCount[-1]==len(templates)):
                break
        templatesCount.append(len(templates))
        print(f'Completed iteration')
        rerun-=1
    classify_end = time()
    print(f'Time for generating extra templates and classifying {rerun_ - rerun} times: {classify_end - classsify_start} seconds!')

    backplot_start = time()
    backplot, min, max, templateMatchingResults = classify.backplotImg(radius, imgs, templates)
    backplot_end = time()
    print(f'Time for backplotting-1 : {backplot_end - backplot_start} seconds!')
    
    sort_start = time()
    picDic = cluster.sortTemplates(imgs, templateMatchingResults, radius, templates)
    sort_end = time()
    print(f'Time for sort : {sort_end - sort_start} seconds!')

    templateClassesMap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    i=1
    for pic in picDic:
        for p in pic:
            templateClassesMap[p["xIndex"]:p["xIndex"]+10,p["yIndex"]:p["yIndex"]+10]=i
        i+=1
    fig = px.imshow(templateClassesMap, color_continuous_scale=px.colors.qualitative.Alphabet)
    plotly.offline.plot(fig, filename='./charts/'+imgName+'-templateClasses.html')

    cluster_start = time()
    centroidDic = cluster.cluster(radius, templates, picDic)
    cluster_end = time()
    print(f'Time for clustering : {cluster_end - cluster_start} seconds!')

    noOfPatchesPerPixel = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    # varianceApproxMap = np.zeros((imgs[0].shape[0], imgs[0].shape[1]))
    for i in range(len(centroidDic)):
        for centroid in centroidDic[i]:
            ids = centroid["id"]
            # varianceApprox = np.zeros((2*radius,2*radius))
            for id in ids:
                pos = picDic[i][id]
                # varianceApprox += (pos["template"]-centroid["centroid"])**2
                noOfPatchesPerPixel[pos["xIndex"]:pos["xIndex"]+2*radius,pos["yIndex"]:pos["yIndex"]+2*radius]+=len(ids)
            # varianceApprox /= len(ids)

            # fig = px.imshow(varianceApprox)
            # plotly.offline.plot(fig, filename='./charts/'+imgName+'-varianceApprox.html')

            # fig = px.imshow(centroid["centroid"])
            # plotly.offline.plot(fig, filename='./charts/'+imgName+'-centroid.html')


            # for id in ids:
            #     pos = picDic[i][id]
            #     varianceApproxMap[pos["xIndex"]:pos["xIndex"]+2*radius,pos["yIndex"]:pos["yIndex"]+2*radius]+=variance


    fig = px.imshow(noOfPatchesPerPixel)
    plotly.offline.plot(fig, filename='./charts/'+imgName+'-noOfPatcherPerPixel.html')

    # fig = px.imshow(varianceApproxMap)
    # plotly.offline.plot(fig, filename='./charts/'+imgName+'-varianceApproxMap.html')

    backplot_start = time()
    backplotFinal, min, max, overlayVariance = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)    
    backplot_end = time()
    print(f'Time for backplotting-2 : {backplot_end - backplot_start} seconds!')

    fig = px.imshow(overlayVariance[0][radius:-radius,radius:-radius])
    plotly.offline.plot(fig, filename='./charts/'+imgName+'-overlayVariance.html')

    for i in range(len(imgs)):
        plt.figure(figsize=(2*15, 2*7)) 
        ax1=plt.subplot(1,2,1)                    
        ax1.imshow(imgs[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax1.set_title('original image')
        ax1.axis('off')
        ax2=plt.subplot(1,2,2)                    
        ax2.imshow(backplotFinal[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax2.set_title('backplot')
        ax2.axis('off')
        #plt.figure(figsize=(15, 12))  
        #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
        #plt.colorbar()
        # plt.show()

    plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/results/'+imgName+'-denoised.png')    

    end = time()
    print(f'Total time: {end - start} seconds!')

    plt.figure(figsize=(20,20))

    img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(backplotFinal[0][radius:-radius,radius:-radius]))))
    ax1=plt.subplot(1,2,2)
    ax1.imshow(img,cmap='gray')
    ax1.axis('off')
    ax1.set_title('FFT of denoised image in')
    img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(imgs[0][radius:-radius,radius:-radius]))))
    ax1=plt.subplot(1,2,1)
    ax1.imshow(img,cmap='gray')
    ax1.axis('off')
    ax1.set_title('FFT of original image')
    plt.show()

    return backplotFinal

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