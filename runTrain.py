import os
import numpy as np
from keras import Input, Model, losses, metrics, layers
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt

def normalization(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I

def load_data(data_dir = 'data/train/'):
    points = []
    freestreamV = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        points.append(data['input'][0][:-2].reshape(101))
        freestreamV.append(data['input'][0][-2:].reshape(2))
        # input.append(data['input'][0].reshape(103))
        target.append(np.array(data['LnD'] * 100).reshape(2))
    return np.asarray(points), np.asarray(freestreamV), np.asarray(target)

point, freestreamVelocity, target = load_data()
print(f'points shape : {point.shape}, velocity shape : {freestreamVelocity.shape}, target shape : {target.shape}')

input_points = Input(shape=(101,), name='points')
input_velocity = Input(shape=(2,), name='velocity')

x1 = layers.Dense(64)(input_points)
x2 = layers.Dense(2)(input_velocity)

x = layers.concatenate([x1, x2])
x = layers.Dense(4, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(8, activation='relu')(x)
x = layers.Dense(4, activation='relu')(x)
coef_output = layers.Dense(2)(x)

model = Model(inputs=[input_points, input_velocity], outputs=coef_output)

model.compile(
    optimizer = SGD(learning_rate=0.00000001),
    loss=losses.MeanAbsoluteError(),
    metrics=[metrics.MeanAbsoluteError()],
    )
dnn_model = model.fit([point, freestreamVelocity], target, 
                    #validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    batch_size=16, epochs=100)

model.save('dnn_model.keras')
plt.plot(dnn_model.history['loss'])
plt.title('loss figure')
plt.xlabel('epochs')
plt.ylabel('MAE %')
plt.savefig('loss figure.png')