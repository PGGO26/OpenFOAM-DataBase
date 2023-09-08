import os
import numpy as np
from keras.models import load_model

def load_data(data_dir = 'data/train/'):
    points = []
    freestreamV = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        points.append(data['input'][0][:-2].reshape(101))
        freestreamV.append(data['input'][0][-2:].reshape(2))
        target.append(np.array(data['LnD'] * 100).reshape(2))
    return np.asarray(points), np.asarray(freestreamV), np.asarray(target)

def normalization(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I

point, freestreamVelocity, target = load_data('data/test/')
print(point.shape, freestreamVelocity.shape)

model = load_model('dnn_model.keras')
results = model.evaluate([point, freestreamVelocity], target, batch_size=16)
print("test results (mse):", results)

predictions = model.predict([point, freestreamVelocity])
mape = np.mean(np.abs((target - predictions) / target)) #mape
print(f'mape : {(mape * 100):.2f} %')
for item in range(len(predictions)):
    mape_each = np.abs((target[item] - predictions[item]) / target[item]) * 100
    print(f'predicitons : {predictions[item] / 100}, test : {target[item] / 100}, mape(%) : {mape_each}')