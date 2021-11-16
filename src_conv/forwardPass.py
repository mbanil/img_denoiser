# import keras.backend as K
# import numpy as np
# from keras import Input, layers
# from keras.models import Model

# def my_filter(shape, dtype=None):
    
#     f = np.array([
#             [[[1]], [[0]], [[-1]]],
#             [[[1]], [[0]], [[-1]]],
#             [[[1]], [[0]], [[-1]]]
#         ])
#     assert f.shape == shape
#     return K.variable(f, dtype='float32')



# # def build_model():
# #     input_tensor = Input(shape=(6,6,1))

# #     x = layers.Conv2D(filters=1, 
# #                       kernel_size = 3,
# #                       kernel_initializer=my_filter,
# #                       strides=2, 
# #                       padding='valid') (input_tensor)

# #     model = Model(inputs=input_tensor, outputs=x)
# #     return model


# import numpy as np
# import keras



# def build_model(templates):
#     templates = np.array(templates)

#     input_tensor = Input(shape=templates.shape)

#     x = layers.Conv2D(filters=1, 
#                       kernel_size = templates.shape[1],
#                       kernel_initializer=templates,
#                       strides=1, 
#                       padding='valid') (input_tensor)

#     model = Model(inputs=input_tensor, outputs=x)
#     return model

# def train_model(imgs, templates):

#     model = build_model(templates)
#     model.trainable = False
#     model.compile(optimizer='rmsprop', loss='mse')
    

#     y = np.random.random((10, 5))
#     model.fit(x, y, epochs=10)

    
import numpy as np
import torch
import torch.nn.functional as F

def build_model(imgs,templates):

    print(torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    for template in templates:
        template -= np.mean(template)

    templates_arr = np.array(templates) 
    imgs_arr = np.array(imgs)

    if(len(templates_arr.shape)==3):
        templates_arr = templates_arr[...,np.newaxis]
        imgs_arr = imgs_arr[...,np.newaxis]

    # if(templates_arr.shape[3]==1):    
    #     # templates_new = np.tile(templates[:,0,:,:], [, len(imgs),templates_newaxis.shape[1],templates_newaxis.shape[2]])
    #     templates_newaxis = templates[:, np.newaxis,:,:]
    #     # templates_new = np.tile(templates_newaxis, [len(imgs),1,1,1])
    #     custom_filter = torch.tensor(templates_newaxis, dtype=torch.float32, device=device)
    #     # print(custom_filter.shape)

    #     imgs_arr_newaxis = imgs_arr[:,np.newaxis,:,:]
    #     x = torch.tensor(imgs_arr_newaxis, dtype=torch.float32, device=device)
    #     # print(x.shape)

    #     output = F.conv2d(x, custom_filter, padding=0, stride=1)
    #     ouput_result = output.cpu().detach().numpy()

    #     # dotproduct_result = torch.dot(torch.tensor(imgs_arr), torch.tensor(templates))
    # else:

    ouput_result = []
    for i in range(templates_arr.shape[3]):
        ouput_result.append(perform_conv(templates_arr[:,:,:,i], imgs_arr[:,:,:,i], device)) 
        
    ouput_result = np.stack(ouput_result)

    # if(len(ouput_result.shape)==4):
    #     ouput_result = ouput_result[np.newaxis,...]

    normalized_result = normalize(ouput_result, imgs_arr, templates_arr)

    normalized_result = np.sum(normalized_result, axis=0)/normalized_result.shape[0]

    return normalized_result


def perform_conv(templates, imgs_arr, device):

    # templates = np.array(templates) 
    # templates_new = np.tile(templates[:,0,:,:], [, len(imgs),templates_newaxis.shape[1],templates_newaxis.shape[2]])
    templates_newaxis = templates[:, np.newaxis,:,:]
    # templates_new = np.tile(templates_newaxis, [len(imgs),1,1,1])
    custom_filter = torch.tensor(templates_newaxis, dtype=torch.float32, device=device)
    # print(custom_filter.shape)

    # imgs_arr = np.array(imgs)
    imgs_arr_newaxis = imgs_arr[:,np.newaxis,:,:]
    x = torch.tensor(imgs_arr_newaxis, dtype=torch.float32, device=device)
    # print(x.shape)

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
