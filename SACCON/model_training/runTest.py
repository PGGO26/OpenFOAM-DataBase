# 2025/05/28 runTest.py code for model testing and predictions plotting

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import NPZDataset, ToTensor, Normalize, Denormalize
from Networks.UNet import UNet
from sklearn.metrics import mean_squared_error

# 參數設定(可調整)
# 決定使用模型的架構
IN_CHANNELS = 2
OUT_CHANNELS = 2
NUM_ADDTIONAL_INTPUTS = 2

# Testing Patterns
PATTERN_NAMES = ["SACCON_Origin"]

# 資料處理路徑
MODEL_DIR = "Networks/"
LOSS_DIR = "Loss/"
TEST_DATASET_DIR = "data/validation/"

OUTPUT_DIR = "results/"
VISUAL_DIR = "visualizations/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_visualization(pattern, loss_dir, output_dir):
    train_losses = np.load(os.path.join(loss_dir, f"{pattern}_train_loss.npy"))
    val_losses = np.load(os.path.join(loss_dir, f"{pattern}_val_loss.npy"))
    epochs = range(len(train_losses))
    
    output_fig_path = os.path.join(output_dir, f"UNet_{pattern}_Loss_over_Epochs.png")
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_losses, label="Training loss", color="blue", marker="")
    plt.semilogy(epochs, val_losses, label="Validation loss", color="red", marker="")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss over Epochs", fontsize=16, y=1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fig_path)
    
    return output_fig_path

def result_plotting(prediction_result_dir, output_dir):
    prediction_pattern = [p for p in os.listdir(prediction_result_dir) if p.endswith("predictions.npz")]
    for pattern in prediction_pattern:
        pattern_base = pattern.replace("_predictions.npz", "")
        pattern_prediction_results = np.load(os.path.join(prediction_result_dir, pattern))
        print(f"For {pattern} errors visualizations.........")
        samples = range(len(pattern_prediction_results['cases']))
        for idx in samples:
            basename = pattern_prediction_results['cases'][idx]
            ground_data = pattern_prediction_results['targets'][idx]
            predict_data = pattern_prediction_results['predictions'][idx]
            mask = np.load("data/ref/Mask.npz")['mask']

            ground_upper_data = ground_data[:, :, 0] + mask
            ground_lower_data = ground_data[:, :, 1] + mask
            predict_upper_data = predict_data[:, :, 0] + mask
            predict_lower_data = predict_data[:, :, 1] + mask

            fig, axes = plt.subplots(2, 3, figsize=(22, 13))
            colors = ["#0000F0", "#0073FF", "#00FDFB", "#00FF8A", "#01FF07", "#89FF00", "#FAFF00", "#FF7900", "#FF0000"]
            n_bins = 50
            cmap_name = "SACCON"
            custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
            
            upper_vmin = np.nanmin([np.nanmin(ground_upper_data), np.nanmin(predict_upper_data)])
            upper_vmax = np.nanmax([np.nanmax(ground_upper_data), np.nanmax(predict_upper_data)])
            lower_vmin = np.nanmin([np.nanmin(ground_lower_data), np.nanmin(predict_lower_data)])
            lower_vmax = np.nanmax([np.nanmax(ground_lower_data), np.nanmax(predict_lower_data)])

            im1 = axes[0, 0].imshow(ground_upper_data, cmap=custom_cmap, vmin=upper_vmin, vmax=upper_vmax, interpolation='nearest')
            cbar1 = fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar1.set_label("Cp", fontsize=18)
            axes[0, 0].set_title("Upper surface - Cp Simulation", fontsize=20, y=1.02)
            axes[0, 0].set_xlabel("X-axis")
            axes[0, 0].set_ylabel("Y-axis")

            im2 = axes[0, 1].imshow(predict_upper_data, cmap=custom_cmap, vmin=upper_vmin, vmax=upper_vmax, interpolation='nearest')
            cbar2 = fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar2.set_label("Cp", fontsize=18)
            axes[0, 1].set_title("Upper surface - Cp Prediction", fontsize=20, y=1.02)
            axes[0, 1].set_xlabel("X-axis")
            axes[0, 1].set_ylabel("Y-axis")

            upper_error = np.abs(predict_upper_data - ground_upper_data)
            im3 = axes[0, 2].imshow(upper_error, cmap='Greens', interpolation='nearest')
            cbar3 = fig.colorbar(im3, ax=axes[0, 2], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar3.set_label("Error", fontsize=18)
            axes[0, 2].set_title('Upper surface - Error', fontsize=20, y=1.02)
            axes[0, 2].set_xlabel('X-axis')
            axes[0, 2].set_ylabel('Y-axis')

            im4 = axes[1, 0].imshow(ground_lower_data, cmap=custom_cmap, vmin=lower_vmin, vmax=lower_vmax, interpolation='nearest')
            cbar4 = fig.colorbar(im4, ax=axes[1, 0], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar4.set_label("Cp", fontsize=18)
            axes[1, 0].set_title("Lower surface - Cp Simulation", fontsize=20, y=1.02)
            axes[1, 0].set_xlabel("X-axis")
            axes[1, 0].set_ylabel("Y-axis")

            im5 = axes[1, 1].imshow(predict_lower_data, cmap=custom_cmap, vmin=lower_vmin, vmax=lower_vmax, interpolation='nearest')
            cbar5 = fig.colorbar(im5, ax=axes[1, 1], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar5.set_label("Cp", fontsize=18)
            axes[1, 1].set_title("Lower surface - Cp Prediction", fontsize=20, y=1.02)
            axes[1, 1].set_xlabel("X-axis")
            axes[1, 1].set_ylabel("Y-axis")

            lower_error = np.abs(predict_lower_data - ground_lower_data)
            im6 = axes[1, 2].imshow(lower_error, cmap='Greens', interpolation='nearest')
            cbar6 = fig.colorbar(im6, ax=axes[1, 2], orientation='vertical', shrink=0.85, aspect=20, pad=0.05)
            cbar6.set_label("Error", fontsize=18)
            axes[1, 2].set_title('Lower surface - Error', fontsize=20, y=1.02)
            axes[1, 2].set_xlabel('X-axis')
            axes[1, 2].set_ylabel('Y-axis')

            fig.suptitle("Comparison of Ground vs Prediction Surface Cp Distribution", fontsize=28)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{pattern_base}_result_{basename}.png")
            print(f"Save {pattern_base}_{basename} results.")
            plt.close("all")


# 載入資料集
dataset = NPZDataset(folder_path=TEST_DATASET_DIR)
normalize = Normalize(dataset, keys=['Thickness', 'Chamber', 'Upper_P', 'Lower_P'])
denormalize = Denormalize(normalize)
test_dataset = NPZDataset(
    folder_path=TEST_DATASET_DIR,
    transform=lambda sample: ToTensor()(normalize(sample))
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 儲存模型誤差
from collections import defaultdict

case_wise_errors = defaultdict(dict)
all_case_names = []

# 逐一測試每個模型
for pattern in PATTERN_NAMES:
    MODEL_PATH = os.path.join(MODEL_DIR, f"{pattern}_UNet.pth")
    model = UNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_additional_inputs=NUM_ADDTIONAL_INTPUTS,
        dropout_rate=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    all_cases = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            thickness = batch['Thickness'].to(DEVICE)
            chamber = batch['Chamber'].to(DEVICE)
            inputs = torch.cat((thickness, chamber), dim=1)

            mach = batch['Mach'].unsqueeze(1).to(DEVICE)
            aoa = batch['AOA'].unsqueeze(1).to(DEVICE)

            outputs = model(inputs, mach, aoa).detach().cpu().numpy()
            targets = torch.cat((batch['Upper_P'], batch['Lower_P']), dim=1).numpy()

            outputs_denorm = np.concatenate([
                denormalize({'Upper_P': outputs[:, 0, :, :]})['Upper_P'][..., None],
                denormalize({'Lower_P': outputs[:, 1, :, :]})['Lower_P'][..., None]
            ], axis=-1)

            targets_denorm = np.concatenate([
                denormalize({'Upper_P': targets[:, 0, :, :]})['Upper_P'][..., None],
                denormalize({'Lower_P': targets[:, 1, :, :]})['Lower_P'][..., None]
            ], axis=-1)

            case_name = f"Mach_{mach.max():.2f}_AOA_{aoa.max():.2f}"
            all_case_names.append(case_name)

            pred = outputs_denorm[0]
            target = targets_denorm[0]

            case_mse = mean_squared_error(target.reshape(-1), pred.reshape(-1))
            case_wise_errors[case_name][f"UNet_{pattern}"] = case_mse

            all_cases.append(case_name)
            all_preds.append(outputs_denorm)
            all_targets.append(targets_denorm)

        # 計算 MSE
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        mse = mean_squared_error(all_targets.reshape(all_targets.shape[0], -1), all_preds.reshape(all_preds.shape[0], -1))
        print(f"UNet {pattern} - MSE: {mse:.4f}")

        # 儲存預測結果
        prediction_output_dir = os.path.join(OUTPUT_DIR, pattern)
        visualization_dir = os.path.join(VISUAL_DIR, pattern)
        os.makedirs(prediction_output_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        
        np.savez(os.path.join(prediction_output_dir, f"UNet_{pattern}_predictions.npz"),
                cases=all_cases,
                predictions=all_preds,
                targets=all_targets)

        # 訓練 Loss 變化圖繪製
        Loss_figure_path = loss_visualization(pattern=pattern, loss_dir=LOSS_DIR, output_dir=visualization_dir)

# 繪製誤差長條圖
# 移除重複的 case 名稱並排序
all_case_names = sorted(list(set(all_case_names)))

# 繪製每個 case 的不同模型誤差（群組長條圖）
x = np.arange(len(all_case_names))
width = 0.35  # 每組 bar 的寬度

plt.figure(figsize=(14, 7))

for i, pattern in enumerate(PATTERN_NAMES):
    model_name = f"UNet_{pattern}"
    y = [case_wise_errors[case].get(model_name, np.nan) for case in all_case_names]
    plt.bar(x + i * width, y, width=width, label=model_name)

plt.xlabel("Test Case", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.title("Case-wise Model Comparison on Validation Set", fontsize=14)
plt.xticks(x + width * (len(PATTERN_NAMES)-1) / 2, all_case_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "casewise_model_comparison_barplot.png"))

# 預測結果繪製
for pattern in PATTERN_NAMES:
    prediction_output_dir = os.path.join(OUTPUT_DIR, pattern)
    visualization_dir = os.path.join(VISUAL_DIR, pattern)
    MAE_RESULT = result_plotting(prediction_result_dir=prediction_output_dir, output_dir=visualization_dir)