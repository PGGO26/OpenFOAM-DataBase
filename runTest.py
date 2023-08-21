import os
import numpy as np
from keras.models import load_model

def load_data(data_dir = 'data/test/'):
    input = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        input.append(np.array(data['output'][:3]))
        target.append(np.array(data['LnD']).reshape(1,2))
    img_input = np.moveaxis(input, 1, 3)
    return normalizeation(np.asarray(img_input)), normalizeation(np.asarray(target))

def normalizeation(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I

input_test, target_test = load_data()

model = load_model('cnn_model.keras')

results = model.evaluate(input_test, target_test, batch_size=3)
print("test loss, test acc:", results)

print("Generate predictions for 3 samples")
predictions = model.predict(input_test[:3])
print(f"predictions shape: {predictions.shape} predictions : {predictions}")
# predictions = model.predict(input_test)
# np.mean(np.abs((target_test - predictions))) #mae
# print(predictions)
# mape = np.mean(np.abs((target_test - predictions) / target_test)) * 100 #mape
# print('mape : ', mape)