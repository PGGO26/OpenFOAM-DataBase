import os
import numpy as np
import pandas as pd
from keras import Input, Model, losses, metrics, layers
from keras.optimizers import Adam, SGD

from dptls import data_process
dptls = data_process()
model_input, model_target = dptls.load_data(dataDir='train_data/',airfoilDir = 'Airfoils/')
print(model_target.shape, model_input.shape)

input_point = Input(shape=(302,),name='Points and Velocity')
x = layers.Dense(256)(input_point)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
cf_output = layers.Dense(102)(x)

model = Model(inputs=input_point, outputs=cf_output)

lr=1e-6
epochs = 20000

model.compile(
    optimizer = SGD(learning_rate=lr),
    loss=losses.MeanAbsoluteError(),
    metrics=[metrics.MeanAbsolutePercentageError()],
    )

model.summary()
dnn_model = model.fit(model_input, model_target,
                      validation_split=0.2, batch_size=64, epochs=epochs)

model.save('temp/dnn_model.keras')

# Save the training progress for plotting the loss vs epochs
loss = dnn_model.history['loss']
val_loss = dnn_model.history['val_loss']
np.save('temp/Training loss data', [loss,[epochs, lr], val_loss])