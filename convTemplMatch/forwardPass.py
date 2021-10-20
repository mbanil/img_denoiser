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

    for template in templates:
        template -= np.mean(template)

    templates = np.array(templates) 
    templates_newaxis = templates[:, np.newaxis,:,:]
    custom_filter = torch.tensor(templates_newaxis)
    # print(custom_filter.shape)

    imgs_arr = np.array(imgs)
    imgs_arr_newaxis = imgs_arr[np.newaxis,...]
    x = torch.tensor(imgs_arr_newaxis)
    # print(x.shape)


    output = F.conv2d(x, custom_filter, padding=0)
    ouput_result = output.cpu().detach().numpy()

    # dotproduct_result = torch.dot(torch.tensor(imgs_arr), torch.tensor(templates))

    return ouput_result




    
    
