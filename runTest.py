import os
import numpy as np
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
test_input, test_target = load_data('test/naca001264.dat/v50/')
print(f'input shape : {test_input.shape}, target shape : {test_target.shape}')

model = load_model('dnn_model.keras')
results = model.evaluate(test_input, test_target, batch_size=16)
print("test results (mse):", results)

predictions = model.predict(test_input)
print(f'predictions shape : {predictions.shape}')

# Plot for different between test dataset and predictions
plotting_index = 0
data_dir = 'test/naca001264.dat/v50/'

plotting_predictions = predictions[plotting_index]
datalst = os.listdir(data_dir)
plotting_data = np.load(data_dir + datalst[plotting_index])
cf = plotting_data['cf'][0].reshape(101)
print(f"cf shape is : {cf.shape}")
x = np.moveaxis(np.load('temp/data0.npy'), 0, 1)[0]
y = np.zeros(101)
for i in range(101):
    y[i] = plotting_data['input'][0][i]
cf_diff = cf - plotting_predictions

plt.scatter(x,y, c=cf_diff)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig('Cf Plotting for test dataset')