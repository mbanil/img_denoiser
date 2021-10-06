import matplotlib.pyplot as plt
import numpy as np
from time import time
import cProfile
import io
import pstats

from src import helperfuncs
# from src import classify
from convTemplMatch import classify
from src import cluster

from convTemplMatch import forwardPass



folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/' # Maxime/' #sample 2/'
imgName = '18_04_27_Thomas_28618_0016.dm3'


def denoise(folderPath, imgName, rerun = 5, templateSize = 23):

    start = time()

    startPosList= [[84,404],[97,404],[88,404]]

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

    
    gen_start = time()
    templates = helperfuncs.generateTemplates(startPosList=startPosList, imgs=imgs, radius=templateSize)
    templates = helperfuncs.findDissimilarTemplates(templates = templates, imgs = imgs, radius = templateSize, minTemplateClasses = NumMainclasses)
    gen_end = time()
    print(f'Time for generating basic templates: {gen_end - gen_start} seconds!')

    rerun_ = rerun
    classsify_start = time()
    while rerun>0:
        templates = classify.tempfuncname(radius=templateSize, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
        rerun-=1
    classify_end = time()
    print(f'Time for generating extra templates and classifying {rerun_} times: {classify_end - classsify_start} seconds!')

    

    # firstImg = result[0,1,:,:]

    # n1_max=1
    # n1=0
    # n2_max=result.shape[1]
    # n2=0
    # plt.figure(figsize=(20, 20*n2_max/n1_max)) 
    # for i in range(result.shape[1]):
    #     n2+=1            
    #     ax1=plt.subplot(n2_max,n1_max,n1*n2_max+n2)
    #     ax1.imshow(result[0,i,:,:],cmap='gray')

    # plt.show()
    # # plt.imshow(firstImg)
    # print("Done")



    # model = build_model()
    # out = model.predict(input_mat)
    # print(out)

    radius = templateSize











    backplot_start = time()
    backplot, min, max, templateMatchingResults = classify.backplotImg(radius, imgs, templates)
    backplot_end = time()
    print(f'Time for backplotting-1 : {backplot_end - backplot_start} seconds!')
    
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

    plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/results/convTemp'+imgName+'-denoised.png')    

    end = time()
    print(f'Total time: {end - start} seconds!')

    plt.figure(figsize=(20,20))

    

    img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(imgs[0][radius:-radius,radius:-radius]))))
    ax1=plt.subplot(1,2,1)
    ax1.imshow(img,cmap='gray')
    ax1.axis('off')
    ax1.set_title('FFT of original image')
    img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(backplotFinal[0][radius:-radius,radius:-radius]))))
    ax1=plt.subplot(1,2,2)
    ax1.imshow(img,cmap='gray')
    ax1.axis('off')
    ax1.set_title('FFT of denoised image in')
    plt.show()

    return backplotFinal

# denoise(folderPath, imgName, rerun = 15, radius=23)
# cProfile.run('denoise(folderPath, imgName, rerun = 15, radius=23)')

pr = cProfile.Profile()
pr.enable()
denoise(folderPath, imgName)
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats("denoise")
print(s.getvalue())