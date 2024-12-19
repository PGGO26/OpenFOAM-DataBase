import matplotlib.pyplot as plt
import re, os
import numpy as np

# 繪製真值圖
def plot_groundTruth(key, npOutput, baseName):
    field = np.copy(npOutput)
    field = np.flipud(field.transpose())

    plt.figure(figsize=(8,6))
    if key.split('_')[-1] == 'Z':
        plt.imshow(field, cmap='Greys', interpolation='nearest')
        plt.colorbar(label='Height')
        plt.title(f"{key.split('_')[0]} surface Height Map")
    elif key.split('_')[-1] == 'P':
        plt.imshow(field, cmap='jet', vmin=-1e4, vmax=1.5e5, interpolation='nearest')
        plt.colorbar(label='Pressure')
        plt.title(f"{key.split('_')[0]} surface pressure distribution")
    else:
        plt.imshow(field, cmap='Greys', interpolation='nearest')
        plt.colorbar(label='Height')
        plt.title(f"{key.split('_')[0]} Mask")

    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(f"plots/{baseName}_{key}.png")

def extract_losses(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    train_losses = []
    val_losses = []

    for line in lines:
        # 提取 epoch
        epoch_match = re.search(r'Epoch (\d+)/\d+, Training', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)
        
        # 提取训练损失
        train_loss_match = re.search(r'Training Loss: ([\d.]+)', line)
        if train_loss_match:
            train_loss = float(train_loss_match.group(1))
            train_losses.append(train_loss)
        
        # 提取验证损失
        val_loss_match = re.search(r'Validation Loss: ([\d.]+)', line)
        if val_loss_match:
            val_loss = float(val_loss_match.group(1))
            val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses

# 绘制损失图表
def plot_losses(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Training and Validation Loss over Epochs')


# log_file = 'Log/runTrain.log'
# epochs, train_losses, val_losses = extract_losses(log_file)
# plot_losses(epochs, train_losses, val_losses)

plotDir = 'DataBase/data/test/'
keyLst = ['Upper_Z','Upper_P','Lower_Z','Lower_P', 'Global_Mask']

for file in os.listdir(plotDir):
    fileName = file.split(".npz")[0]
    data_path = os.path.join(plotDir + file)
    data = np.load(data_path)
    for key in keyLst:
        npOutput = data[key]
        print(f"{fileName}_{key} shape : {np.shape(npOutput)[0]}")
        plot_groundTruth(key, npOutput, baseName=fileName)
