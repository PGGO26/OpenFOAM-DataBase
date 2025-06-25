import os

LOG_DIR = "Log/"
LOSS_DIR = "Loss/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)


def train_pattern(pattern_name, train_dir, val_dir, model_path, log_path, loss_dir,
                  para_config, base_model_path=None, freeze_encoder=False):
    
    import logging
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import StepLR

    from utils import NPZDataset, AugmentedDataset, ToTensor, Normalize, GeometricTransformations
    from utils import EarlyStopping
    from Networks.UNet import UNet
    import numpy as np
    import os

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Logging
    logging.basicConfig(
        filename=log_path, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info(f"Start training for pattern: {pattern_name}")
    logging.info(f"Using device: {device}")

    # Unpack hyperparameters
    IN_Channels = para_config['in_channels']
    OUT_Channels = para_config['out_channels']
    NUM_Addtional_Inputs = para_config['num_additional_inputs']
    DROPOUT_RATE = para_config['dropout_rate']
    batch_SIZE = para_config['batch_size']
    learning_RATE = para_config['learning_rate']
    step_SIZE = para_config['step_size']
    GAMMA = para_config['gamma']
    num_epochs = para_config['num_epochs']
    NUM_Patience = para_config['patience']
    DELTA = para_config['delta']

    # Dataset preparation
    geo_transform = GeometricTransformations(angles=[90, 180, 270])
    train_base = NPZDataset(train_dir)
    normalize = Normalize(train_base, keys=['Thickness', 'Chamber', 'Upper_P', 'Lower_P'])
    train_dataset = NPZDataset(train_dir, transform=lambda s: ToTensor()(normalize(s)))
    val_base = NPZDataset(val_dir)
    val_dataset = NPZDataset(val_dir, transform=lambda s: ToTensor()(normalize(s)))

    train_aug = AugmentedDataset(train_dataset, geo_transform)
    val_aug = AugmentedDataset(val_dataset, geo_transform)
    train_loader = DataLoader(train_aug, batch_size=batch_SIZE, shuffle=True)
    val_loader = DataLoader(val_aug, batch_size=batch_SIZE, shuffle=False)

    # Model, loss, optimizer
    model = UNet(IN_Channels, OUT_Channels, NUM_Addtional_Inputs, DROPOUT_RATE).to(device)

    if base_model_path is not None and os.path.exists(base_model_path):
        logging.info(f"[{pattern_name}] Loading base model from {base_model_path}")
        model.load_state_dict(torch.load(base_model_path))
        if freeze_encoder:
            logging.info(f"[{pattern_name}] Freezing encoder layers")
            for name, param in model.named_parameters():
                if 'enc' in name:
                    param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100.0 * trainable_params / total_params
    logging.info(f"[{pattern_name}] Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_RATE)
    scheduler = StepLR(optimizer, step_size=step_SIZE, gamma=GAMMA)
    early_stopper = EarlyStopping(patience=NUM_Patience, delta=DELTA, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            inputs = torch.cat((batch['data']['Thickness'].to(device), batch['data']['Chamber'].to(device)), dim=1)
            targets = torch.cat((batch['data']['Upper_P'].to(device), batch['data']['Lower_P'].to(device)), dim=1)
            mach = batch['data']['Mach'].unsqueeze(1).to(device)
            aoa = batch['data']['AOA'].unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs, mach, aoa)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.cat((batch['data']['Thickness'].to(device), batch['data']['Chamber'].to(device)), dim=1)
                targets = torch.cat((batch['data']['Upper_P'].to(device), batch['data']['Lower_P'].to(device)), dim=1)
                mach = batch['data']['Mach'].unsqueeze(1).to(device)
                aoa = batch['data']['AOA'].unsqueeze(1).to(device)
                outputs = model(inputs, mach, aoa)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)
        val_losses.append(avg_val)

        early_stopper(avg_val, model)
        if early_stopper.early_stop:
            logging.info(f"[{pattern_name}] Early stopping triggered at epoch {epoch+1}")
            assert early_stopper.best_model_wts is not None, "best_model_wts is None. EarlyStopping failed to store best weights."
            model.load_state_dict(early_stopper.best_model_wts)
            break

        logging.info(f"[{pattern_name}] Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}, Best Val Loss = {early_stopper.best_loss:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), model_path)
    logging.info(f"[{pattern_name}] Model saved to {model_path}")
    np.save(os.path.join(loss_dir, f"{pattern_name}_train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(loss_dir, f"{pattern_name}_val_loss.npy"), np.array(val_losses))
    logging.info(f"[{pattern_name}] Loss saved.")


para_config = {
    'in_channels': 2,
    'out_channels': 2,
    'num_additional_inputs': 2,
    'dropout_rate':0.1,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'step_size': 150,
    'gamma': 0.7,
    'num_epochs': 2000,
    'patience': 200,
    'delta': 1e-3
}

train_pattern(
    pattern_name="SACCON_Origin",
    train_dir="data/train_all/",
    val_dir="data/validation/",
    model_path="Networks/SACCON_Origin_UNet.pth",
    log_path="Log/SACCON_Origin_train.log",
    loss_dir="Loss/",
    para_config=para_config,
    base_model_path=None,
    freeze_encoder=False
)

# train_pattern(
#     pattern_name="B",
#     train_dir="data/train_B/",
#     val_dir="data/validation/",
#     model_path="Networks/UNet_B.pth",
#     log_path="Log/train_B.log",
#     loss_dir="Loss/",
#     para_config=para_config,
#     base_model_path="Networks/Basic_UNet.pth",
#     freeze_encoder=True
# )
