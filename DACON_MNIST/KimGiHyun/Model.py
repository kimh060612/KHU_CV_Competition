import tensorflow as tf
from tensorflow import keras as tfk

class AuxNetLayer(tfk.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        

    def build(self):
        pass

    def call(self):
        pass

    