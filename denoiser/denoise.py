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

import cProfile
import io
import pstats

from src import helperfuncs
from src_parallel import classify_parallel
from src_conv import classify_conv
from src import classify
from src import cluster


def intialize_patches(image, radius):

    initial_patch_locations = []
    no_of_initial_patches = np.max(image.shape)//100
    if no_of_initial_patches < 2:
        no_of_initial_patches = 2
    for i in range(no_of_initial_patches):
        lower_limit = int((i/no_of_initial_patches)*(image.shape[1]-2*radius))
        upper_limit = int(((i+1)/no_of_initial_patches)
                          * (image.shape[1]-2*radius))
        rand_X = randint(lower_limit, upper_limit)
        initial_patch_locations.append(
            [rand_X, randint(0, image.shape[2]-2*radius)])

    return initial_patch_locations

def run_profiler(image, min_patches_per_class=5, max_patches_per_class=100, iteration_counter=15, patch_size=48, termination_number=3, analyze=False, clustering_factor=2.7):

    pr = cProfile.Profile()
    pr.enable()
    denoised_image  = denoiser(image, min_patches_per_class, max_patches_per_class, iteration_counter, patch_size, termination_number, analyze, clustering_factor)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats("denoise")
    print(s.getvalue())

    return denoised_image


def denoiser(image, min_patches_per_class=5, max_patches_per_class=100, iteration_counter=15, patch_size=48, termination_number=3, analyze=False, clustering_factor=2.7):
    # image shape is (channels/n,height,width)
    # patch size is for example (48,48)

    radius = patch_size//2
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]

    intial_patch_locations = intialize_patches(image=image, radius=radius)
    print(intial_patch_locations)

    max_patches_per_class = max_patches_per_class * \
        int(np.ceil(np.sqrt(image.shape[0])))

    templates = helperfuncs.generateTemplates(
        intial_patch_locations=intial_patch_locations, image=image, radius=radius)

    templatesCount = []

    if torch.cuda.is_available() or image.shape[0] == 1:

        while iteration_counter > 0:
            templates = classify_conv.tempfuncname(radius=radius, image=image, templates=templates,
                                                   maxNumberInClass=max_patches_per_class, minNumberInClass=min_patches_per_class)
            if (len(templatesCount) > termination_number):
                if (templatesCount[-1] == len(templates)):
                    if (templatesCount[-1] == templatesCount[-2]):
                        break
            templatesCount.append(len(templates))
            iteration_counter -= 1
        backplot, min, max, templateMatchingResults = classify_conv.backplotImg(
            radius, image, templates)

    else:

        pool = mp.Pool(mp.cpu_count())
        while iteration_counter > 0:
            templates = classify_parallel.tempfuncname(radius=radius, image=image, templates=templates,
                                                       maxNumberInClass=max_patches_per_class, minNumberInClass=min_patches_per_class, pool=pool)
            if (len(templatesCount) > termination_number):
                if (templatesCount[-1] == len(templates)):
                    if (templatesCount[-1] == templatesCount[-2]):
                        break
            templatesCount.append(len(templates))
            print(f'Completed iteration')
            iteration_counter -= 1
        backplot, min, max, templateMatchingResults = classify_parallel.backplotImg(
            radius, image, templates, pool)
        pool.close()

    picDic = cluster.sortTemplates(
        image, templateMatchingResults, radius, templates)

    if (analyze == True):

        for i in range(len(image)):
            plt.figure(figsize=(2*15, 2*7))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(image[i], cmap=plt.cm.gray)
            ax1.set_title('original image')
            ax1.axis('off')
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(backplot[i], cmap=plt.cm.gray)
            ax2.set_title('backplot')
            ax2.axis('off')
            plt.show()

        templateClassesMap = np.zeros((image.shape[1], image.shape[2]))
        i = 1
        for pic in picDic:
            for p in pic:
                templateClassesMap[p["xIndex"]:p["xIndex"] +
                                   10, p["yIndex"]:p["yIndex"]+10] = i
            i += 1
        fig = px.imshow(templateClassesMap,
                        color_continuous_scale=px.colors.qualitative.Alphabet)

    centroidDic = cluster.cluster(radius, templates, picDic, clustering_factor)

    if (analyze == True):
        noOfPatchesPerPixel = np.zeros((image[0].shape[0], image[0].shape[1]))
        for i in range(len(centroidDic)):
            for centroid in centroidDic[i]:
                ids = centroid["id"]
                for id in ids:
                    pos = picDic[i][id]
                    noOfPatchesPerPixel[pos["xIndex"]:pos["xIndex"]+2*radius,
                                        pos["yIndex"]:pos["yIndex"]+2*radius] += len(ids)

        fig = px.imshow(noOfPatchesPerPixel)

    backplotFinal, min, max, overlayVariance = cluster.backplotFinal(
        centroidDic, picDic, image, radius, templateMatchingResults)
    backplotFinal = helperfuncs.adjustEdges(backplotFinal, image)

    if (analyze == True):
        fig = px.imshow(
            np.sqrt(overlayVariance[0][radius:-radius, radius:-radius]))
        fig = px.imshow(backplotFinal[0][radius:-radius, radius:-radius])
        fig = px.imshow((backplotFinal[0][radius:-radius, radius:-
                        radius] - image[0][radius:-radius, radius:-radius])**2)

        for i in range(len(image)):
            plt.figure(figsize=(20, 20))
            img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image[i]))))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(img, cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of original image')
            img = np.log(np.abs(np.fft.fftshift(
                np.fft.fft2(backplotFinal[i]))))
            ax1 = plt.subplot(1, 2, 2)
            ax1.imshow(img, cmap='gray')
            ax1.axis('off')
            ax1.set_title('FFT of denoised image in')

            plt.show()


    return backplotFinal


