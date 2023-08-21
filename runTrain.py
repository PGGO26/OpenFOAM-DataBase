
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
        target.append(np.array(data['LnD']).reshape(1,2))
    img_input = np.moveaxis(input, 1, 3)
    return normalizeation(np.asarray(img_input)), normalizeation(np.asarray(target))

def normalizeation(I):
    I = ((I - I.min()) / (I.max() - I.min()))
    return I


input, target = load_data() # argument 為訓練輸入的 data 資料夾，預設為 data/train/
print(f'input shape is : {input.shape}  target shape is : {target.shape}')

input_shape = (256, 256, 3)
model = Sequential([
    Conv2D(64, 3, input_shape=input_shape, padding='same',
           activation='relu', strides=1),
    MaxPooling2D(pool_size=(2, 2), strides=1),
#     Dropout(0.2),
    # Conv2D(128, 3, input_shape=input_shape, padding='same', activation='relu', strides=1),
    # MaxPooling2D(pool_size=(2, 2), strides=1),
#     Dropout(0.2),
    Flatten(),
#     Dropout(0.5),
    # Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='relu')
])

model.compile(optimizer = SGD(learning_rate=0.01),loss = 'mse',metrics=['mse', 'mape'])
cnn = model.fit(input, target, 
                    #validation_data=(test_images, test_labels),
                    #verbose=2,callbacks=[earlyStop],
                    batch_size=3, epochs=2)

model.save('cnn_model.keras')

