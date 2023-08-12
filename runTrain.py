
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD

def load_data(data_dir = 'data/train/'):
    input = []
    target = []
    for file in os.listdir(data_dir):
        data = np.load(data_dir + file)
        input.append(np.array(data['output'][:3]))
        target.append(np.array(data['LnD'][0]))
    img_input = np.moveaxis(input, 1, 3)
    return np.asarray(img_input), np.asarray(target)

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 200
    return I.astype(np.uint8)


input, target = load_data() # argument 為訓練輸入的 data 資料夾，預設為 data/train/
print(f'input shape is : {input.shape}  target shape is : {target.shape}')

input_shape = (256, 256, 3)
model = Sequential([
    Conv2D(64, 3, input_shape=input_shape, padding='same',
           activation='relu', strides=1),
    # Conv2D(3, 16,  padding='same',
    #        activation='relu', strides=2),
    MaxPooling2D(pool_size=(2, 2), strides=1),
#     Dropout(0.2),
    # Conv2D(128, 3, input_shape=input_shape, padding='same',
    #        activation='relu', strides=1),
    # Conv2D(16, 64,padding='same',
    #        activation='relu', strides=2),
    # MaxPooling2D(pool_size=(2, 2), strides=1),
#     Dropout(0.2),
    Flatten(),
#     Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(2, activation='relu')
])

model.compile(optimizer = SGD(learning_rate=0.1),loss = 'mse',metrics=['mse', 'mape'])
cnn = model.fit(input, target, 
                    #validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    batch_size=50, epochs=5)


# predictions = model.predict(testX)
# np.mean(np.abs((testY - predictions))) #mae
# np.mean(np.abs((testY - predictions) / testY)) * 100 #mape