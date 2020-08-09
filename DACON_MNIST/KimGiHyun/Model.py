import tensorflow as tf
from tensorflow import keras as tfk

class ResNetLayer(tfk.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def call(self):
        pass

class ResNetModel(tfk.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self):
        pass
    