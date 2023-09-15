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
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
cf_output = layers.Dense(101)(x)

model = Model(inputs=input_PnV, outputs=cf_output)

model.compile(
    optimizer = SGD(learning_rate=0.0005),
    loss=losses.MeanAbsoluteError(),
    metrics=['accuracy'],
    )

model.summary()
dnn_model = model.fit(train_input, train_target,
                      validation_split=0.2, batch_size=16, epochs=800)

model.save('dnn_model.keras')

# # */*/*/*/
# acc = dnn_model.history['accuracy']
# val_acc = dnn_model.history['val_accuracy']
# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
# plt.title('Traininig And Validation')
# plt.xlabel('epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('Training and Validation.png')
# # */*/*/*/d

loss = dnn_model.history['loss']
val_loss = dnn_model.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs[200:], loss[200:], 'b-', label='Training loss')
plt.plot(epochs[200:], val_loss[200:], 'r-', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Training and Validation loss')