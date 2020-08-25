import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import pandas as pd

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

DEFAULT_FUNCTION_KEY = "serving_default"
loaded_model = tf.saved_model.load("model")
inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]
loaded_test_data = np.load("ImageTest.npy")
loaded_test_letter_data = np.load("TargetLetterTest.npy")

length_data = loaded_test_data.shape[0]

batch_size = 64

predict = []

for batch_index in range(length_data//batch_size):
    currBatch = batch_index*batch_size
    nextBatch = (batch_index + 1)*batch_size
    predict.append(inference_func(tf.convert_to_tensor(loaded_test_data[currBatch : nextBatch], dtype=tf.float32)))

answer = []

for pred in predict:
    Out_Tensor = pred['output_1']
    OutNumpy = Out_Tensor.numpy()
    for i in range(OutNumpy.shape[0]):
        Probabilty_Mass = OutNumpy[i]
        answer.append(np.argmax(Probabilty_Mass))

answer = np.array(answer)

print(answer)
print(answer.shape)

with open("submission.csv","w") as File:
    File.write("id,digit\n")
    index = 2049
    for i in range(answer.shape[0]):
        File.write(str(index+i)+","+str(answer[i])+"\n")

print("done")