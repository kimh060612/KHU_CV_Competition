import tensorflow as tf
import numpy as np
from tensorflow import keras

loaded_model = keras.models.load_model("model_ResNet")
loaded_test_data = np.load("ImageTest.npy")

predict = loaded_model.predict(loaded_test_data)
print(predict)

print(predict.shape)

