import logging
import matplotlib.pyplot as plt
import torch

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from utils import NPZDataset, Normalize, ToTensor, GeometricTransformations, ApplyFilters
from Models.UNet import UNet

# Configure logging
logging.basicConfig(filename='Log/runTrain.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w')
train_losses = []
val_losses = []

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set model parameters
batch_SIZE = 32
learning_RATE = 1e-4
step_SIZE = 150
GAMMA = 0.7

IN_Channels = 1
OUT_Channels = 2
NUM_Addtional_Inputs = 2
Base_Channels = 16

# Load training data
train_data_dir = "DataBase/data/train/"
logging.info("Loading training data.")
transform = transforms.Compose([ToTensor(), Normalize()])
train_dataset = NPZDataset(train_data_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_SIZE, shuffle=True)

# Load validation data
val_data_dir = "DataBase/data/validation/"
logging.info("Loading validation data.")
val_dataset = NPZDataset(val_data_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_SIZE, shuffle=False)

# Number of additional variables (Mach and AOA)
num_additional_inputs = 2

# Initialize model, loss function, and optimizer
# model = UNet(in_channels=IN_Channels, out_channels=OUT_Channels, num_additional_inputs=NUM_Addtional_Inputs,
#             sample_channels=sample_channels, bottle_channels=bottle_channels).to(device)
model = UNet(in_channels=IN_Channels, out_channels=OUT_Channels, num_additional_inputs=NUM_Addtional_Inputs, base_channels=Base_Channels).to(device)
criterion = nn.MSELoss().to(device)
# criterion = WeightedMSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_RATE)

# Initialize learning rate scheduler
scheduler = StepLR(optimizer, step_size=step_SIZE, gamma=GAMMA)

# Calculating params
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {int(total_params // 1e6)},{str(total_params)[-7:-4]},{str(total_params)[-4:-1]}')

# Training loop
num_epochs = 900
logging.info("Start training.")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_dataloader:
        upper_z = batch['Upper_Z'].to(device)
        lower_z = batch['Lower_Z'].to(device)
        # inputs = torch.cat((upper_z, lower_z), dim=1)
        inputs = upper_z
        upper_p = batch['Upper_P'].to(device)
        lower_p = batch['Lower_P'].to(device)
        targets = torch.cat((upper_p, lower_p), dim=1)
        mach = batch['Mach'].unsqueeze(1).to(device)
        aoa = batch['AOA'].unsqueeze(1).to(device)

        # Forward pass
        outputs = model(inputs, mach, aoa)
        # loss = criterion(outputs, targets, weight)
        loss = criterion(outputs, targets)
        
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 計算訓練損失
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Synchronize GPU
    torch.cuda.synchronize()

    logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_dataloader)}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            upper_z = batch['Upper_Z'].to(device)
            lower_z = batch['Lower_Z'].to(device)
            # inputs = torch.cat((upper_z, lower_z), dim=1)
            inputs = upper_z
            upper_p = batch['Upper_P'].to(device)
            lower_p = batch['Lower_P'].to(device)
            targets = torch.cat((upper_p, lower_p), dim=1)
            mach = batch['Mach'].unsqueeze(1).to(device)
            aoa = batch['AOA'].unsqueeze(1).to(device)

            outputs = model(inputs, mach, aoa)
            # loss = criterion(outputs, targets, weight)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)  # 保存驗證損失

    # Synchronize GPU
    torch.cuda.synchronize()

    logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader)}")
    
    # 每次 scheduler 更新時繪製損失圖表
    if (epoch + 1) % step_SIZE == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', marker='')
        plt.plot(val_losses, label='Validation Loss', color='red', marker='')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve up to Epoch {epoch+1}')
        plt.grid()
        plt.savefig(f'plots/loss_epoch_{epoch+1}.png')
        plt.close()

    # Step the scheduler
    scheduler.step()

# Save the model
torch.save(model.state_dict(), f'Models/UNet.pth')
logging.info("Model saved.")

