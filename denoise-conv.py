import matplotlib.pyplot as plt
import numpy as np
from time import time
import cProfile
import io
import pstats
import plotly
import plotly.express as px

from src import helperfuncs
# from src import classify
from convTemplMatch import classify
from src import cluster

from convTemplMatch import forwardPass



# folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-2/' # Maxime/' #sample 2/'
folderPath = 'C:/My Documents/TUD-MCL/Semester 4/Thesis/Implementation/Data/Dataset-1/' 
imgName = '18_04_27_Thomas_28618_0017.dm3'
# imgName = 'Stack_zeolite4NaAF__111_001_1-10.tif'


def denoise(folderPath, imgName, rerun = 15, templateSize = 23):

    start = time()

    # startPosList= [[25,150],[75,250]]

    radius= templateSize

    startPosList= [[84-radius,104-radius],[97-radius,204-radius],[88-radius,154-radius]]


    load_start = time()
    imgs = helperfuncs.loadData(folderPath=folderPath, fileName=imgName)
    load_end = time()
    print(f'Time for loading: {load_end - load_start} seconds!')

    NumMainclasses=4
    MinNumberInClass=4
    # MaxNumberInClass=100
    MaxNumberInClass=100*int(np.ceil(np.sqrt(len(imgs))))

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

    # n1_max=1
    # n1=0
    # n2_max=len(templates)
    # n2=0
    # plt.figure(figsize=(5, 5*n2_max/n1_max)) 
    # for i in range(len(templates)):
    #     n2+=1            
    #     ax1=plt.subplot(n2_max,n1_max,n1*n2_max+n2)
    #     ax1.imshow(templates[i],cmap='gray')

    # plt.show()
    
    # templates = helperfuncs.findDissimilarTemplates(templates = templates, imgs = imgs, radius = templateSize, minTemplateClasses = NumMainclasses)
    gen_end = time()
    print(f'Time for generating basic templates: {gen_end - gen_start} seconds!')

    rerun_ = rerun
    classsify_start = time()
    while rerun>0:
        templates = classify.tempfuncname(radius=templateSize, imgs=imgs, templates=templates, maxNumberInClass=MaxNumberInClass, minNumberInClass=MinNumberInClass)
        rerun-=1
    classify_end = time()
    print(f'Time for generating extra templates and classifying {rerun_} times: {classify_end - classsify_start} seconds!')

    # result = forwardPass.build_model(imgs, templates)

    # fig = px.imshow(result[0,0,:,:])
    # plotly.offline.plot(fig, filename='./'+imgName+'-conv.html')

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

    # idx = (-result[0,0,:,:].flatten()).argsort()
    # idxd=np.unravel_index(idx,result[0,0,:,:].shape)

    # similarTemplates = []

    
    # n1_max=1
    # n1=0
    # n2_max=10
    # n2=0
    # plt.figure(figsize=(10, 10*n2_max/n1_max)) 

    # for i in range(10):
    #     similarTemplates.append(imgs[0][idxd[0][i]:idxd[0][i]+2*templateSize,idxd[1][i]:idxd[1][i]+2*templateSize])
    #     n2+=1            
    #     ax1=plt.subplot(n2_max,n1_max,n1*n2_max+n2)
    #     ax1.imshow(similarTemplates[i],cmap='gray')
    




    # model = build_model()
    # out = model.predict(input_mat)
    # print(out)

    radius = templateSize











    backplot_start = time()
    backplot, min, max, templateMatchingResults = classify.backplotImg(radius, imgs, templates)
    backplot_end = time()
    print(f'Time for backplotting-1 : {backplot_end - backplot_start} seconds!')



    for i in range(len(imgs)):
        plt.figure(figsize=(2*15, 2*7)) 
        ax1=plt.subplot(1,2,1)                    
        ax1.imshow(backplot[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax1.set_title('backplot')
        ax1.axis('off')
        ax2=plt.subplot(1,2,2)                    
        ax2.imshow(imgs[i][radius:-radius,radius:-radius],cmap=plt.cm.gray,vmin=min[i],vmax=max[i])
        ax2.set_title('original image')
        ax2.axis('off')
        #plt.figure(figsize=(15, 12))  
        #plt.imshow(overlayclass[Mode][myindex],cmap=plt.cm.gist_rainbow)
        #plt.colorbar()
        # plt.show()
    
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
    plotly.offline.plot(fig, filename='./charts/'+imgName+'-convtemplateClasses.html')


    cluster_start = time()
    centroidDic = cluster.cluster(radius, templates, picDic)
    cluster_end = time()
    print(f'Time for clustering : {cluster_end - cluster_start} seconds!')

    backplot_start = time()
    backplotFinal, min, max, overlayVariance = cluster.backplotFinal(centroidDic, picDic, imgs, radius, templateMatchingResults)    
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
        # plt.show()

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