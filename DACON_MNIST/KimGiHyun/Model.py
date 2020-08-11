import tensorflow as tf
from tensorflow import keras as tfk

# ResNet50 Structure

class ResidualBlock(tfk.layers.Layer):
    def __init__(self, InputChannel, OutputChannel, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

        self.IsSkipConnection = False
        self.Batch1 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv1 = tfk.layers.Conv2D(filters=OutputChannel//2, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU1 = tfk.layers.LeakyReLU()
        self.Batch2 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv2 = tfk.layers.Conv2D(filters=OutputChannel//2, kernel_size=(3, 3), strides=(1, 1))
        self.LeakyReLU2 = tfk.layers.LeakyReLU()
        self.Batch3 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv3 = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU3 = tfk.layers.LeakyReLU()

        # Skip Connection
        self.SkipConnection = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(3, 3), strides=(1, 1))
        self.LeakyReLUSkip = tfk.layers.LeakyReLU()
        if (not InputChannel == OutputChannel):
            self.SkipConnection = True

    def call(self, Input):
        Skip = Input
        if self.IsSkipConnection :
            Skip = self.SkipConnection(Skip)
            Skip = self.LeakyReLUSkip(Skip)
        Z = Input
        Z = self.Batch1(Z)
        Z = self.conv1(Z)
        Z = self.LeakyReLU1(Z)
        Z = self.Batch2(Z)
        Z = self.conv2(Z)
        Z = self.LeakyReLU2(Z)
        Z = self.Batch3(Z)
        Z = self.conv3(Z)
        Z = self.LeakyReLU3(Z)
        return Z + Skip

class ResNetModel(tfk.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.Padding1 = tfk.layers.ZeroPadding2D(padding=(3, 3))
        self.conv1 = tfk.layers.Conv2D(filters=32, kernel_size=(5,5), stride=(2,2))
        self.Batch1 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.Activation1 = tfk.layers.ReLU()
        self.Padding2 = tfk.layers.ZeroPadding2D(padding=(1,1))

        self.ResLayer1 = ResidualBlock(32, 64)
        self.ResLayer2 = ResidualBlock(64, 64)

    def call(self, Input):
        Z = Input
        Z = self.Padding1(Z)
        Z = self.conv1(Z)
        Z = self.Batch1(Z)
        Z = self.Activation1(Z)
        Z = self.Padding2(Z)
        Z = self.ResLayer1(Z)
        Z = self.ResLayer2(Z)
        
        return Z
