import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model

from dptls import data_process
dptls = data_process()
test_input, test_target = dptls.load_data(dataDir='test_dataset/data/',airfoilDir='test_dataset/Airfoils/')
print(test_input.shape, test_target.shape)

model = load_model('temp/dnn_model.keras')
results = model.evaluate(test_input, test_target, batch_size=16)
print("test results (mse):", results)

# prediction
predic_input, predic_target = dptls.load_data(dataDir='test_dataset/prediction_dataset/data/',airfoilDir='test_dataset/prediction_dataset/Airfoils/')
print(predic_input.shape, predic_target.shape)

predictions = model.predict(predic_input)
print(f'predictions shape : {predictions.shape}')
np.save('temp/predictions', predictions)