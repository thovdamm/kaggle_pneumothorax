import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def unet(depth = 3,filters = 16):
    i = layers.Input(shape=(1024,1024,61),dtype = np.float32)
    d=i
    c=[]
    for j in range(depth):
        c_,d = down_block(d,filters = filters*(2**j))
        c.append(c_)

    x = layers.Conv2D(filters*(2**(depth-1)),kernel_size=3,activation='relu',padding='same')(d)

    for j in reversed(range(depth)):
        x = up_block(x,c[j],filters = filters*(2**j))

    o = layers.Conv2D(filters = 1,kernel_size = 3 ,padding='same')(x)

    return tf.keras.Model(inputs = i,outputs = o)

def down_block(x,filters,kernel=3,activation='relu',name_scope=None):
    c = layers.Conv2D(filters = filters,kernel_size = kernel,activation=activation,padding='same')(x)
    c = layers.Conv2D(filters = filters,kernel_size = kernel,activation=activation,padding='same')(c)
    d = layers.MaxPooling2D(pool_size=(2,2))(c)
    return c,d

def up_block(x,y,filters,kernel=3,activation='relu',name_scope=None):
    #x = layers.Conv2D(filters = filters//2,kernel_size = kernel,activation=activation,padding='same')(x)    
    u = layers.UpSampling2D(size=(2,2))(x)
    c = layers.Concatenate(axis=-1)([u,y])
    c = layers.Conv2D(filters = filters,kernel_size = kernel,activation=activation,padding='same')(c)
    #c = layers.Conv2D(filters = filters,kernel_size = kernel,activation=activation,padding='same')(c)
    return c


