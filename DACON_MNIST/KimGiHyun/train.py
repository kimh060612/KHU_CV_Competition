import tensorflow as tf
import matplotlib.pyplot as plt
from Model import ResidualBlock, ResNetModel
import numpy as np
import os

EPOCHS = 100

if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    Train_data = np.load("Image.npy")
    Train_target_digit = np.load("TargetDigit.npy")
    Train_target_letter = np.load("TargetLetter.npy")   
    #Input = tf.keras.Input((28, 28, 1), dtpye="float32", name="input_1")
    ResNet = ResNetModel(output_dim=10)
    ResNet.build(input_shape=(1, 28, 28, 1))
    ResNet.summary()

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()

    ResNet.compile(optimizer, CrossEntropyLoss)
    ResNet.fit(Train_data, Train_target_digit, epochs=EPOCHS, batch_size=64)

    ResNet.save("model_ResNet")