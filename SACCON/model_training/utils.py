import os, copy
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class NPZDataset(Dataset):
    """
    加載 npz 文件並提供數據處理（正規化、幾何變換等）的接口。
    """
    def __init__(self, folder_path, transform=None, augmentations=None):
        self.folder_path = folder_path
        self.npz_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')
        ]
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        # 加載 npz 文件
        npz_file = self.npz_files[idx]
        npz_data = np.load(npz_file)
        sample = {
            'Thickness': npz_data['Map'][0],
            'Chamber': npz_data['Map'][1],
            'Upper_P': npz_data['Cp'][0],
            'Lower_P': npz_data['Cp'][1],
        }

        # 從文件名中提取 Mach 和 AOA
        filename = os.path.basename(npz_file)
        parts = filename.split('_')
        sample.update({
            'Mach': float(parts[-3]),
            'AOA': float(parts[-1].replace('.npz', '')),
            'baseName': filename
        })

        # 應用數據增強和轉換
        if self.augmentations:
            sample = self.augmentations(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    """將數據轉換為 Tensor 格式"""
    def __call__(self, sample):
        processed_sample = {}
        for key, value in sample.items():
            if key in ['Thickness', 'Chamber', 'Upper_P', 'Lower_P']:
            # if key in ['Thickness', 'Chamber', 'Upper_P']:
                processed_sample[key] = value.clone().detach().unsqueeze(0).to(dtype=torch.float32)
            else:
                # 保留非數據鍵值
                processed_sample[key] = value
        return processed_sample


class Normalize:
    """對數據進行 L2 正規化處理"""
    def __init__(self, dataset, keys):
        """
        :param dataset: 數據集，用於計算全局 mean 和 std
        :param keys: 需要正規化的鍵值
        """
        self.keys = keys
        self.means = {}
        self.stds = {}

        # 計算全局的 mean 和 std
        all_data = {key: [] for key in keys}
        for sample in dataset:
            for key in keys:
                all_data[key].append(torch.tensor(sample[key], dtype=torch.float32))

        for key in keys:
            concatenated = torch.cat([d.flatten() for d in all_data[key]])
            self.means[key] = torch.mean(concatenated)
            self.stds[key] = torch.std(concatenated)

    def __call__(self, sample):
        normalized_sample = {}
        for key in self.keys:
            tensor = torch.tensor(sample[key], dtype=torch.float32)
            normalized_sample[key] = (tensor - self.means[key]) / self.stds[key]

        # 保留其他鍵值
        normalized_sample.update({k: sample[k] for k in sample if k not in normalized_sample})
        return normalized_sample


class Denormalize:
    """對數據進行去正規化處理"""
    def __init__(self, normalize_instance):
        """
        :param normalize_instance: Normalize 的實例，用於提取 mean 和 std
        """
        self.means = normalize_instance.means
        self.stds = normalize_instance.stds

    def __call__(self, sample):
        denormalized_sample = {}
        for key, value in sample.items():
            if key in self.means and key in self.stds:
                tensor = torch.tensor(value, dtype=torch.float32)
                denormalized_sample[key] = tensor * self.stds[key] + self.means[key]
            else:
                # 保留其他未正規化的鍵
                denormalized_sample[key] = value

        return denormalized_sample


class GeometricTransformations:
    """對數據應用幾何變換（旋轉、平移、縮放）"""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, sample):
        transformed_samples = []
        for angle in self.angles:
            transformed_sample = {}
            for key in ['Thickness', 'Chamber', 'Upper_P', 'Lower_P']:
            # for key in ['Thickness', 'Chamber', 'Upper_P']:
                tensor = sample[key].clone().detach().unsqueeze(0).to(dtype=torch.float32)
                transformed_sample[key] = F.affine(tensor, angle=angle, translate=(0, 0), scale=1.0, shear=0).squeeze(0)
            transformed_sample.update({k: sample[k] for k in sample if k not in transformed_sample})
            transformed_samples.append(transformed_sample)
        return transformed_samples


class AugmentedDataset(Dataset):
    """
    將原始數據和幾何變換後的數據結合為一個數據集。
    """
    def __init__(self, base_dataset, geometric_transformations):
        self.base_dataset = base_dataset
        self.geometric_transformations = geometric_transformations

    def __len__(self):
        return len(self.base_dataset) * (len(self.geometric_transformations.angles) + 1)

    def __getitem__(self, idx):
        original_idx = idx // (len(self.geometric_transformations.angles) + 1)
        transform_idx = idx % (len(self.geometric_transformations.angles) + 1)

        sample = self.base_dataset[original_idx]

        if transform_idx == 0:
            return {"data": sample, "label": f"{sample['baseName']}_original"}
        else:
            transformed_sample = self.geometric_transformations(sample)[transform_idx - 1]
            return {"data": transformed_sample, "label": f"{sample['baseName']}_transformed_{transform_idx}"}


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
