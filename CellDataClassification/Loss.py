from tensorflow import keras
import tensorflow as tf
import numpy as np

class AsymetricLoss(keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        
    
    def call(self, y_true, y_pred):
        pass