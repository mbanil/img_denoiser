  
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

# import plotly.express as px
# import plotly

def build_model(image,templates):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # template_old = deepcopy(templates)

    # normalize templates
    for template in templates:
        if(len(template.shape)==3):
            for i in range(template.shape[2]):
                template[:,:,i] -= np.mean(template[:,:,i])
        else:
            template -= np.mean(template)

    templates_arr = np.array(templates) 

    if(len(templates_arr.shape)==3):
        templates_arr = templates_arr[...,np.newaxis]
        imgs_arr = image[...,np.newaxis]


    # check what is i in this loop
    ouput_result = []
    for i in range(templates_arr.shape[3]):
        ouput_result.append(perform_conv(templates_arr[:,:,:,i], imgs_arr[:,:,:,i], device)) 
        
    ouput_result = np.stack(ouput_result)

    normalized_result = normalize(ouput_result, imgs_arr, templates_arr)

    normalized_result = np.sum(normalized_result, axis=0)/normalized_result.shape[0]

    return normalized_result


def perform_conv(templates, imgs_arr, device):

    templates_newaxis = templates[:, np.newaxis,:,:]
    custom_filter = torch.tensor(templates_newaxis, dtype=torch.float32, device=device)

    imgs_arr_newaxis = imgs_arr[:,np.newaxis,:,:]
    x = torch.tensor(imgs_arr_newaxis, dtype=torch.float32, device=device)

    output = F.conv2d(x, custom_filter, padding=0, stride=1)
    ouput_result = output.cpu().detach().numpy()

    return ouput_result


def normalize(convResults, imgs, templates):

    for channel in range(convResults.shape[0]):
        for i in range(convResults[channel].shape[0]):
            convImg = np.sum(imgs[i,:,:,channel]*imgs[i,:,:,channel])            
            for j in range(convResults[channel,i,:,:,:].shape[0]):
                convTemplate = np.sum(templates[j,:,:,channel]*templates[j,:,:,channel])
                convResults[channel,i,j,:,:] /= np.sqrt(convImg*convTemplate)

    return convResults
