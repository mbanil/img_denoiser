import skimage
from copy import deepcopy
import numpy as np
import pickle
import os
from pathlib import Path
import cv2

def extractFeatures(imgs, stride, templateSize):
    
    surf = cv2.xfeatures2d.SURF_create()

    for img in imgs:
        for i in range(0, img.shape[0]-templateSize, stride):
            for j in range(0, img.shape[1]-templateSize, stride):
                template = img[i:i+templateSize, j:j+templateSize]
                kp, des = surf.detectAndCompute(template, None)
                print(des)
                


def createTemplates(imgs, stride, templateSize, templatesPath):

    allTemplates = []

    for img in imgs:
        for i in range(0, img.shape[0]-templateSize, stride):
            for j in range(0, img.shape[1]-templateSize, stride):
                temp = img[i:i+templateSize, j:j+templateSize]
                allTemplates.append((temp))

    allTemplates=np.array(allTemplates)    
    
    with open(templatesPath, 'wb') as f:
        pickle.dump(allTemplates, f)

    return allTemplates

    
def loadTemplates(imgs, stride, templateSize, templatesPath):

    if templatesPath.is_file():
        file = open(templatesPath, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        data = createTemplates(imgs, stride, templateSize, templatesPath)

    return data

