import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model

def load_data(data_dir = 'data/train/'):
    input = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        input.append(data['input'][0].reshape(103))
        target.append(data['cf'][0].reshape(101))
    return np.asarray(input), np.asarray(target)

def normalization(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I

# Test
test_dir = 'test/ah94156.dat/v40/'
test_input, test_target = load_data(test_dir)
print(f'input shape : {test_input.shape}, target shape : {test_target.shape}')

model = load_model('dnn_model.keras')
results = model.evaluate(test_input, test_target, batch_size=16)
print("test results (mse):", results)

predictions = model.predict(test_input)
print(f'predictions shape : {predictions.shape}')
np.save('temp/predictions', predictions)