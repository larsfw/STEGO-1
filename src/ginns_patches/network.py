
import numpy as np
import os
import sys

for p in sys.path:
    for root, dirs, files in os.walk(p):
        for d in dirs:
            if d.endswith("giNNs"):
                path_lib = os.path.join(root, d) + '\\libs\\'
#                print(os.path.join(root, d))
            else:
                path_lib = '.\\ginns\\libs\\'
print(path_lib)
#sys.path.append(path_lib + 'SegNet-master\\')

sys.path.append(path_lib)
from SegNet.model import segnet
from unet.model import unet
from DeepLabv3plus import Deeplabv3
#sys.path.append(path_lib + 'unet-master\\')


from tensorflow.keras.layers import Reshape, Input, Multiply, Concatenate
from tensorflow.keras import Model #keras.models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%

def imageGenerator(X_data, Y_mask, inSeed, batchSize, dictionary):
    """ Clip an image/tensor to patches with specific dimention, using padding at the edges.
        The number of patches in x and y dimention and the width and length of the padded area is 
        saved in metaInfo.
        Input:
        param X_data  numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        param Y_mask  numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, 1] 
        param inSeed  random seed for  data augmentation
        param dictionary  dictionary with tags for data augmentation
        param batchSize  batch size
        Output:
        yield data_with_mask  arrays with instance from X_data and Y_mask
    """
    datagen_data = ImageDataGenerator(**dictionary)
    #dictionary_mask = dictionary
    if 'brightness_range' in dictionary.keys():
        print('Delete brightness')
        dictionary.pop('brightness_range')
    
    datagen_mask = ImageDataGenerator(**dictionary)
    Y_dummies = np.zeros(np.shape(X_data)[0])
    image_generator_data = datagen_data.flow(
            X_data,
            Y_dummies,
            seed = inSeed,
            batch_size = batchSize,
            save_prefix = 'image')
    image_generator_mask = datagen_mask.flow(
            Y_mask,
            Y_dummies,
            seed = inSeed,
            batch_size = batchSize,
            save_prefix = 'mask')
    while True:
            data = image_generator_data.next()
            data = data[0]
            mask = image_generator_mask.next()
            mask = mask[0]
#            shape2 = np.shape(mask)
            data_with_mask = ([data],mask)
            yield data_with_mask
        
def imageMaskGenerator(X_data, Y_mask, inSeed, batchSize, dictionary, weights = False):
    """ Clip an image/tensor to patches with specific dimention, using padding at the edges.
        The number of patches in x and y dimention and the width and length of the padded area is 
        saved in metaInfo.
        Input:
        param X_data  numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        param Y_mask  numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, 1] 
        param inSeed  random seed for  data augmentation
        param dictionary  dictionary with tags for data augmentation
        param batchSize  batch size
        Output:
        yield data_with_mask  arrays with instance from X_data and Y_mask
    """

    datagen_data = ImageDataGenerator(**dictionary)

    image_generator_data = datagen_data.flow(
            X_data,
            Y_mask,
            seed = inSeed,
            batch_size=batchSize,
            save_prefix = 'image')

    while True:
        a = image_generator_data.next()
        im1 = a[0]
        mask = a[1]
#        print(np.shape(im1))
#        print(np.shape(mask))
        X_roi = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2] ,1])
        for ii in range(0, mask.shape[0]-1):       
                t = mask[ii][:,:,0] != 0
                for jj in range(1,mask.shape[-1]):
                        t = np.logical_or((mask[ii][:,:,jj] != 0), t)
                X_roi[ii] = t[:,:,np.newaxis]

        mask_shape = np.shape(mask)
        mask = np.reshape(mask, [mask_shape[0], mask_shape[1]*mask_shape[2], mask_shape[3]])
        gt = mask
        if (weights):
                w = ((mask[:,:,0] != 0) & (mask[:,:,0] != 0))
                w = w.astype(float)
                res = (im1,gt,w)
        else:
                res = ([im1, X_roi], gt)
        yield res

def deeplabv3plus_mask(weights='pascal_voc', input_shape = [(256, 256, 3), (256, 256, 1)], num_classes = 2, backbone = 'xception', OS = 16,  alpha=1.):
    input_tensor = Input(shape = input_shape[0])
    model = Deeplabv3(weights = weights, input_tensor = input_tensor, input_shape = input_shape[0], classes = num_classes, backbone = backbone, OS = OS, alpha = alpha)
    input_mask = Input(shape = input_shape[1])
    mask = input_mask
    for ii in range(0,num_classes-1):
        mask = Concatenate()([mask, mask])
    x = Multiply()([model.output, mask])
    model = Model(inputs = [input_tensor, input_mask] , outputs = x)
    return model

def segNet_mask(input_shape = [(256, 256, 3), (256, 256, 1)], num_classes = 2, kernel = 3, pool_size = (2, 2), output_mode="softmax"):
    model = segnet(input_shape = input_shape[0], n_labels = num_classes, kernel = kernel, pool_size = pool_size, output_mode = output_mode)
    input_mask = Input(shape = input_shape[1])
    mask = input_mask
    for ii in range(0,num_classes-1):
        mask = Concatenate()([mask,input_mask])
    mask = Reshape(target_shape = (input_shape[0][0]* input_shape[0][1], num_classes))(mask)
    x = Multiply()([model.output, mask])
    model = Model(inputs = [model.input, input_mask] , outputs = x)
    return model

def unet_mask(input_shape = [(256, 256, 3), (256, 256, 1)], num_classes = 2, activation="softmax", pretrained_weights = None):
    model = unet(input_size = input_shape[0], num_classes = num_classes, activation = activation, pretrained_weights = pretrained_weights)
    input_mask = Input(shape = input_shape[1])
    mask = input_mask
    for ii in range(0,num_classes-1):
        mask = Concatenate()([mask,mask])
    x = Multiply()([model.output, mask])
    model = Model(inputs = [model.input, input_mask] , outputs = x)
    return model