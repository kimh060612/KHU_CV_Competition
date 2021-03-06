import tensorflow as tf
import matplotlib.pyplot as plt
from Model import ResidualBlock, ResNetModel
import numpy as np
import os
from sklearn.model_selection import KFold

EPOCHS = 350

if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    Train_data = np.load("Image.npy")
    Train_target_digit = np.load("TargetDigit.npy")
    Train_target_letter = np.load("TargetLetter.npy")
    #Input = tf.keras.Input((28, 28, 1), name="input_1")
    ResNet = ResNetModel(output_dim1=10, output_dim2=26)
    ResNet.build(input_shape=(1, 28, 28, 1))
    ResNet.summary()

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()

    ResNet.compile(optimizer, loss=[CrossEntropyLoss, CrossEntropyLoss], loss_weights=[0.7, 0.3], metrics=['accuracy', 'accuracy'])

    k = 8
    kfold = KFold(n_splits=k, random_state=777)
    for train_index, validation_index in kfold.split(Train_data):
        train_x, val_x = Train_data[train_index], Train_data[validation_index]
        train_letter, val_letter = Train_target_letter[train_index], Train_target_letter[validation_index]
        train_digit, val_digit = Train_target_digit[train_index], Train_target_digit[validation_index]

        ResNet.fit(train_x, [train_digit, train_letter], epochs=EPOCHS, batch_size=64, validation_data=(val_x, [val_digit, val_letter]))


    tf.saved_model.save(ResNet, "model")