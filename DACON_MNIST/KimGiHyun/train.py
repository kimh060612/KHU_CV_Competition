import tensorflow as tf
import matplotlib.pyplot as plt
from Model import ResidualBlock, ResNetModel
import numpy as np

EPOCHS = 100

if __name__ == "__main__":
    Train_data = np.load("Image.npy")
    Train_target_digit = np.load("TargetDigit.npy")
    Train_target_letter = np.load("TargetLetter.npy")   
    #Input = tf.keras.Input((28, 28, 1), dtpye="float32", name="input_1")
    ResNet = ResNetModel()
    ResNet.summary()

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()

    ResNet.compile(optimizer, CrossEntropyLoss)
    ResNet.fit(Train_data, Train_target_digit, epochs=EPOCHS, batch_size=64)

