import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from Models.UNet import UNet

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
In_Channels = 1
Out_Channels = 1
Num_Additional_Inputs = 2
Base_Channels = 16

# Load trained model
model = UNet(In_Channels, Out_Channels, Num_Additional_Inputs, base_channels=Base_Channels).to(device)
model.load_state_dict(torch.load("Models/UNet_16.pth"))
model.eval()

def visualize_upper_prediction(predict, baseName, cmap='jet'):
    predict_image = np.flipud(predict.transpose())
    plt.figure(figsize=(8, 6))
    plt.imshow(predict_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Prediction value')
    plt.title(f"Upper_P Prediction for Sample: {baseName}")
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(f'plots/{baseName}.png')
    plt.close()

# Prepare prediction inputs : "Upper_Z", "Mach", "AOA"
data = np.load("DataBase/data/test/AileM6_Origin.msh_0.83_6.npz")
upper_z = data['Upper_Z']
mach = 0.83
aoa_Lst = range(0, 16)

# Transform to tensor and normalize
upper_z = torch.tensor(upper_z, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
upper_z = upper_z.unsqueeze(1)  # Add channel dimension
upper_z = ((upper_z - torch.mean(upper_z)) / torch.std(upper_z)).to(device)

with torch.no_grad():
    mach_tensor = torch.tensor(mach, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    mach = mach_tensor.unsqueeze(1).to(device)  # Add channel dimension

    for aoa in aoa_Lst:
        baseName = "AileM6_Origin_" + str(aoa)
        aoa_tensor = torch.tensor(aoa, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        aoa = aoa_tensor.unsqueeze(1).to(device)  # Add channel dimension
        
        output = model(upper_z, mach, aoa)

        prediction = output.cpu().numpy()
        visualize_upper_prediction(prediction[0,0], baseName)
