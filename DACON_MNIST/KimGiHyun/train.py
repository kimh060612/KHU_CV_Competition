import tensorflow as tf
import matplotlib.pyplot as plt
from Model import ResidualBlock, ResNetModel
import numpy as np

if __name__ == "__main__":
    Train_data = np.load("Image.npy")
    Train_target_digit = np.load("TargetDigit.npy")
    Train_target_letter = np.load("TargetLetter.npy")   
    Input = tf.keras.Input((28, 28, 1), dtpye="float32", name="input_1")
    ResNet = ResNetModel()
    X = ResNet(Input)
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    output = tf.keras.layers.Dense(10, activation="softmax")(X)
    
    ResidualNet = tf.keras.models.Model(Input, output)
    ResidualNet.summary()

    

