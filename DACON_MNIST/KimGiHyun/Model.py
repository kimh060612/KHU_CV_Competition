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
        self.conv2 = tfk.layers.Conv2D(filters=OutputChannel//2, kernel_size=(3, 3), strides=(1, 1), padding="SAME")
        self.LeakyReLU2 = tfk.layers.LeakyReLU()
        self.Batch3 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.conv3 = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLU3 = tfk.layers.LeakyReLU()

        # Skip Connection
        self.SkipConnection = tfk.layers.Conv2D(filters=OutputChannel, kernel_size=(1, 1), strides=(1, 1))
        self.LeakyReLUSkip = tfk.layers.LeakyReLU()
        if InputChannel != OutputChannel:
            self.IsSkipConnection = True

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

class SqeeuzeNetBlock(tfk.layers.Layer):
    def __init__(self, output_dim, ratio, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.output_dim = output_dim
        self.GAP = tfk.layers.GlobalAveragePooling2D()
        self.excitation1 = tfk.layers.Dense(units=self.output_dim/ratio, kernel_initializer="he_normal")
        self.ReLU1 = tfk.layers.ReLU()
        self.excitation2 = tfk.layers.Dense(units=self.output_dim, kernel_initializer="he_normal")
    
    def call(self, Input):
        Z = Input
        Z = self.GAP(Z)
        Z = self.excitation1(Z)
        Z = self.ReLU1(Z)
        Z = self.excitation2(Z)
        Z = tf.keras.activations.sigmoid(Z)
        Z = tf.reshape(Z, [-1, 1,1,self.output_dim])
        return Input * Z

class ResNetModel(tfk.Model):
    def __init__(self, output_dim1, output_dim2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.Padding1 = tfk.layers.ZeroPadding2D(padding=(3, 3))
        self.conv1 = tfk.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(2,2))
        self.Batch1 = tfk.layers.BatchNormalization(momentum=0.99, epsilon= 0.001)
        self.Activation1 = tfk.layers.ReLU()
        self.Padding2 = tfk.layers.ZeroPadding2D(padding=(1,1))

        self.PreResLayer1 = ResidualBlock(32, 64)
        self.PreSEBlock1 = SqeeuzeNetBlock(64, 2)
        self.PreResLayer2 = ResidualBlock(64, 64)
        self.PreSEBlock2 = SqeeuzeNetBlock(64, 2)
        
        self.ResLayer1 = ResidualBlock(64, 64)
        self.SEBlock1 = SqeeuzeNetBlock(64, 4)
        self.ResLayer2 = ResidualBlock(64, 64)
        self.SEBlock2 = SqeeuzeNetBlock(64, 4)
        self.ResLayer3 = ResidualBlock(64, 128)
        self.conv_mid1 = tfk.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2))
        #self.Drop1 = tfk.layers.Dropout(0.2)
        self.ResLayer4 = ResidualBlock(128, 128)
        self.SEBlock3 = SqeeuzeNetBlock(128, 4)
        self.ResLayer5 = ResidualBlock(128, 64)
        self.SEBlock4 = SqeeuzeNetBlock(64, 4)
        self.ResLayer6 = ResidualBlock(64, 64)
        self.ResLayer7 = ResidualBlock(64, 64)
        self.ResLayer8 = ResidualBlock(64, 64)

        self.AuxLayer1 = ResidualBlock(64, 128)
        self.SEBlockAux1 = SqeeuzeNetBlock(128, 4)
        self.AuxLayer2 = ResidualBlock(128, 128)
        self.SEBlockAux2 = SqeeuzeNetBlock(128, 4)
        self.conv_aux1 = tfk.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2))
        #self.Drop2 = tfk.layers.Dropout(0.2)
        self.AuxLayer3 = ResidualBlock(128, 128)
        self.AuxLayer4 = ResidualBlock(128, 64)
        self.SEBlockAux3 = SqeeuzeNetBlock(64, 4)
        self.AuxLayer5 = ResidualBlock(64, 64)
        self.SEBlockAux4 = SqeeuzeNetBlock(64, 4)
        self.AuxLayer6 = ResidualBlock(64, 64)
        self.AuxLayer7 = ResidualBlock(64, 64)
        self.AuxLayer8 = ResidualBlock(64, 64)
        self.AuxDense = tfk.layers.Dense(output_dim2, activation="softmax")

        self.GlobalAvgPool = tfk.layers.GlobalAveragePooling2D()
        self.OutputDense = tfk.layers.Dense(output_dim1, activation="softmax")

    def call(self, Input):
        Z = Input
        Z = self.Padding1(Z)
        Z = self.conv1(Z)
        Z = self.Batch1(Z)
        Z = self.Activation1(Z)
        Z = self.Padding2(Z)
        Z = self.PreResLayer1(Z)
        Z = self.PreSEBlock1(Z)
        Z = self.PreResLayer2(Z)
        Z = self.PreSEBlock2(Z)
        Aux = Z
        Z = self.ResLayer1(Z)
        Z = self.SEBlock1(Z)
        Z = self.ResLayer2(Z)
        Z = self.SEBlock2(Z)
        Z = self.ResLayer3(Z)
        Z = self.conv_mid1(Z)
        #Z = self.Drop1(Z)
        Z = self.ResLayer4(Z)
        Z = self.SEBlock3(Z)
        Z = self.ResLayer5(Z)
        Z = self.SEBlock4(Z)
        Z = self.ResLayer6(Z)
        Z = self.ResLayer7(Z)
        Z = self.ResLayer8(Z)
        Z = self.GlobalAvgPool(Z)
        Z = self.OutputDense(Z)
        Aux = self.AuxLayer1(Aux)
        Aux = self.SEBlockAux1(Aux)
        Aux = self.AuxLayer2(Aux)
        Aux = self.SEBlockAux2(Aux)
        Aux = self.conv_aux1(Aux)
        #Aux = self.Drop2(Aux)
        Aux = self.AuxLayer3(Aux)
        Aux = self.AuxLayer4(Aux)
        Aux = self.SEBlockAux3(Aux)
        Aux = self.AuxLayer5(Aux)
        Aux = self.SEBlockAux4(Aux)
        Aux = self.AuxLayer6(Aux)
        Aux = self.AuxLayer7(Aux)
        Aux = self.AuxLayer8(Aux)
        Aux = self.GlobalAvgPool(Aux)
        Aux = self.AuxDense(Aux)
        return Z, Aux

class XecptionNet(tfk.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        