import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import NPZDataset, ToTensor, Normalize, Denormalize
from Models.UNet import UNet

# Set parameters
IN_Channels = 1
OUT_Channels = 2
NUM_Addtional_Inputs = 2
Base_Channels = 16
test_data_dir = "DataBase/data/test/"
model_path = f"Models/UNet.pth"

# Load testing data
transform = transforms.Compose([ToTensor(),Normalize()])
test_dataset = NPZDataset(test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = UNet(IN_Channels, OUT_Channels, NUM_Addtional_Inputs, base_channels=Base_Channels)
model.load_state_dict(torch.load(model_path))
model.eval()

def visaulize_result(predict, target, basename, surface, cmap='jet'):
    title_font = {'fontsize': 11, 'fontweight': 'bold'}
    
    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    
    # Adjust subplot positions manually using set_position
    axs[0].set_position([0.08, 0.05, 0.205, 0.85])  # First subplot position
    axs[1].set_position([0.35, 0.05, 0.225, 0.85])  # Manually adjust axs[1] position to reduce distance with axs[0]
    axs[2].set_position([0.7, 0.05, 0.225, 0.85])  # Manually adjust axs[2] position to increase distance from axs[1]

    predict_image = np.flipud(predict.transpose())
    target_image = np.flipud(target.transpose())
    p_inf = 98858.97
    error = np.abs(predict - target) / p_inf
    error_image = np.flipud(error.transpose())
    mach = basename.split("_")[-2]
    aoa = basename.split("_")[-1]

    axs[0].imshow(target_image, cmap=cmap, vmin=0, vmax=1.5e5, interpolation='nearest')
    axs[0].set_title(f"Ground truth", fontdict=title_font)
    axs[0].set_xlabel("X pixel")
    axs[0].set_ylabel("Y pixel")

    im1 = axs[1].imshow(predict_image, cmap=cmap, vmin=0, vmax=1.5e5, interpolation='nearest')
    axs[1].set_title(f"Prediction", fontdict=title_font)
    axs[1].set_xlabel("X pixel")

    axs[2].imshow(error_image, cmap='Greens', interpolation='nearest')
    axs[2].set_title(f"Calculated error", fontdict=title_font)
    axs[2].set_xlabel("X pixel")

    # Create a divider for colorbars between axs[1] and axs[2]
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)  # Custom colorbar for axs[1]

    # Create a second colorbar for axs[2]
    divider2 = make_axes_locatable(axs[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)  # Custom colorbar for axs[2]

    # Add the colorbars to the appropriate positions
    fig.colorbar(im1, cax=cax1, orientation='vertical', label='Pressure (Pa)')
    fig.colorbar(axs[2].images[0], cax=cax2, orientation='vertical', label='Error')

    plt.suptitle(f"Result of {surface} surface pressure for case Mach: {mach} AOA: {aoa}",
                 fontsize=20, fontweight='bold', x=0.5, y=0.95)
    
    plt.savefig(f"plots/Result_{surface}_p__Mach_{mach}_AOA_{aoa}.png")
    plt.close()

def calculate_and_visualize_error(predict, target, basename, surface, cmap='Greens'):
    p_inf = 98858.97
    error = np.abs(predict - target) / p_inf
    error_image = np.flipud(error.transpose())
    mach = basename.split("_")[-2]
    aoa = basename.split("_")[-1]
    plt.figure(figsize=(8, 6))
    plt.imshow(error_image, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Prediction Error')
    plt.title(f"Calculated {surface} surface error for case Mach : {mach} AOA : {aoa}")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.savefig(f"plots/Error_{surface}_p_Mach_{mach}_AOA_{aoa}.png")
    plt.close()

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputs = batch['Upper_Z']
        upper_p = batch['Upper_P']
        lower_p = batch['Lower_P']
        mach = batch['Mach'].unsqueeze(1)
        aoa = batch['AOA'].unsqueeze(1)
        fileName = batch['baseName'][0]
        baseName = fileName.split(".npz")[0]

        # Load normalization factors
        mean_upper_p = batch['mean_upper_p']
        mean_lower_p = batch['mean_lower_p']
        std_upper_p = batch['std_upper_p']
        std_lower_p = batch['std_lower_p']

        # Initialize Denormalize class
        denormalize = Denormalize(mean_upper_p=mean_upper_p, std_upper_p=std_upper_p,
                                    mean_lower_p=mean_lower_p, std_lower_p=std_lower_p)
        
        # Forward pass
        outputs = model(inputs, mach, aoa)
        upper_p_output = outputs[:, 0, :, :].unsqueeze(1)
        lower_p_output = outputs[:, 1, :, :].unsqueeze(1)
        print("Upper p output size : ", upper_p_output.size())
        print("Lower p output size : ", lower_p_output.size())

        # Apply denormalization to upper_p_output
        denorm_outputs = denormalize({'Upper_P':upper_p_output, 'Lower_P':lower_p_output})
        denorm_targets = denormalize({'Upper_P':upper_p, 'Lower_P':lower_p})

        denorm_upper_output = denorm_outputs['Upper_P'].squeeze().cpu().numpy()
        denorm_lower_output = denorm_outputs['Lower_P'].squeeze().cpu().numpy()
        denorm_upper_target = denorm_targets['Upper_P'].squeeze().cpu().numpy()
        denorm_lower_target = denorm_targets['Lower_P'].squeeze().cpu().numpy()

        visaulize_result(denorm_upper_output, denorm_upper_target, baseName, surface='upper')
        visaulize_result(denorm_lower_output, denorm_lower_target, baseName, surface='lower')

        # calculate_and_visualize_error(denorm_upper_output, denorm_upper_target, baseName, surface='upper')
        # calculate_and_visualize_error(denorm_lower_output, denorm_lower_target, baseName, surface='lower')

        print(f"Prediction and error images for sample {baseName} saved.")
