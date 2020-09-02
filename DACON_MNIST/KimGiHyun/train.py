import tensorflow as tf
import matplotlib.pyplot as plt
from Model import ResidualBlock, ResNetModel
import numpy as np
import os

EPOCHS = 800

if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    Train_data = np.load("Image.npy")
    Train_target_digit = np.load("TargetDigit.npy")
    Train_target_letter = np.load("TargetLetter.npy")
    #Input = tf.keras.Input((28, 28, 1), name="input_1")
    ResNet = ResNetModel(output_dim1=10, output_dim2=26)
    ResNet.build(input_shape=(1, 28, 28, 1))
    ResNet.summary()

    optimizer = tf.keras.optimizers.Adam(lr=0.005)
    CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()

    ResNet.compile(optimizer, loss=[CrossEntropyLoss, CrossEntropyLoss], loss_weights=[0.5, 0.5], metrics=['accuracy', 'accuracy'])
    ResNet.fit(Train_data, [Train_target_digit, Train_target_letter], epochs=EPOCHS, batch_size=64)

    tf.saved_model.save(ResNet, "model")

