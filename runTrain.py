import os
import numpy as np
from keras import Input, Model, losses, metrics, layers
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt

def normalization(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I

def load_data(data_dir = 'data/train/'):
    input = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        input.append(data['input'][0].reshape(103))
        target.append(data['cf'][0].reshape(101))
    return np.asarray(input), np.asarray(target)

train_input, train_target = load_data('train/')
print(f'input shape : {train_input.shape}, target shape : {train_target.shape}')

input_PnV = Input(shape=(103,), name='Points and Velocity')

x = layers.Dense(128)(input_PnV)
x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
cf_output = layers.Dense(101)(x)

model = Model(inputs=input_PnV, outputs=cf_output)

model.compile(
    optimizer = SGD(learning_rate=0.003),
    loss=losses.MeanAbsoluteError(),
    metrics=[metrics.MeanAbsolutePercentageError()],
    )

model.summary()
dnn_model = model.fit(train_input, train_target,
                      validation_split=0.2, batch_size=32, epochs=16000)

model.save('dnn_model.keras')

# Save the training progress for plotting the loss vs epochs
loss = dnn_model.history['loss']
val_loss = dnn_model.history['val_loss']
np.save('temp/Training loss data', [loss, val_loss])