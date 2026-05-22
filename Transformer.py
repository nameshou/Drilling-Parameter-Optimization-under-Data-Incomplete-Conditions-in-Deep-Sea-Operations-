import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
import math
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

class DrillingDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_level=0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx].clone()

        if self.augment:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise

        return x, y

def load_and_preprocess_data(file_path, target_col='drilling_time', seq_length=10, augment_factor=1):

    df = pd.read_excel(file_path)

    df = df.dropna()
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    y_log = np.log1p(y)


    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_length:i])
        y_seq.append(y_log[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)



    if augment_factor > 1:
        X_aug, y_aug = [], []
        for _ in range(augment_factor - 1):
            noise = np.random.normal(0, 0.01, X_seq.shape)
            X_noisy = X_seq + noise
            X_aug.append(X_noisy)
            y_aug.append(y_seq)
        X_seq = np.concatenate([X_seq] + X_aug, axis=0)
        y_seq = np.concatenate([y_seq] + y_aug, axis=0)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler_X, feature_cols

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.input_projection(src) * (self.d_model ** 0.5)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        pred = self.output_layer(output)
        return pred.squeeze(-1)

def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    losses, mae_scores = [], []
    mae_criterion = nn.L1Loss()

    for epoch in range(epochs):
        total_loss, total_mae, num_batches = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mae = mae_criterion(outputs, batch_y)
            total_mae += mae.item()
            num_batches += 1

        avg_loss = total_loss / len(train_loader)
        avg_mae = total_mae / num_batches
        losses.append(avg_loss)
        mae_scores.append(avg_mae)

        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train_MAE: {avg_mae:.4f}')
    return losses, mae_scores

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions_raw = np.expm1(predictions)
    actuals_raw = np.expm1(actuals)

    r2 = r2_score(actuals_raw, predictions_raw)
    mae = mean_absolute_error(actuals_raw, predictions_raw)
    rmse = np.sqrt(mean_squared_error(actuals_raw, predictions_raw))
    return actuals_raw, predictions_raw, r2, mae, rmse

def train_main():

    import os
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.xlsx")
    seq_length = 20
    batch_size = 32
    epochs = 500
    lr = 0.001
    augment_factor = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_preprocess_data(
        file_path, seq_length=seq_length, augment_factor=augment_factor
    )

    train_dataset = DrillingDataset(X_train, y_train, augment=True, noise_level=0.01)
    test_dataset = DrillingDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    model = TransformerRegressor(
        input_dim=input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses, train_maes = train_model(model, train_loader, criterion, optimizer, device, epochs)

    actuals, predictions, r2, test_mae, rmse = evaluate_model(model, test_loader, device)
    print(f"R^2:  {r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


    torch.save(model.state_dict(), 'transformer_model.pth')
    joblib.dump(scaler, 'scaler_X.pkl')
    with open('feature_cols.txt', 'w', encoding='utf-8') as f:
        f.write(','.join(feature_cols))

    plot_training_curves(train_losses, train_maes, actuals, predictions, r2, test_mae)

def plot_training_curves(train_losses, train_maes, actuals, predictions, r2, test_mae):

    errors = actuals - predictions
    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(2, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss Curve (MSE)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.6)

    ax2 = plt.subplot(2, 3, 2)
    plt.plot(train_maes, 'g-', linewidth=2)
    plt.title('Training MAE Trend', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.6)

    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(actuals, predictions, alpha=0.6, s=30, c='steelblue', edgecolor='k', linewidth=0.5)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Scatter (R^2={r2:.3f})')
    plt.axis('equal')

    ax4 = plt.subplot(2, 3, 4)
    sample_count = min(100, len(actuals))
    x = np.arange(sample_count)
    plt.plot(x, actuals[:sample_count], 'o-', label='Actual', markersize=6)
    plt.plot(x, predictions[:sample_count], 's-', label='Predicted', markersize=6)
    plt.xlabel('Sample Index')
    plt.ylabel('Drilling Time')
    plt.title('First 100 Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    plt.scatter(predictions, errors, alpha=0.6, s=30, c='orange', edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.6)

    ax6 = plt.subplot(2, 3, 6)
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(errors)
    x_range = np.linspace(errors.min(), errors.max(), 1000)
    plt.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
    plt.axvline(x=0, color='k', linestyle='--', lw=1)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()

    plt.suptitle(f'Transformer Performance (Test MAE: {test_mae:.2f})', fontsize=16)
    plt.tight_layout()
    plt.savefig('transformer_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_from_list(data_list, model_path='transformer_model.pth', scaler_path='scaler_X.pkl',
                      feature_cols_path='feature_cols.txt', seq_length=20, has_target=True):

    df = pd.DataFrame(data_list)

    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        saved_features = f.read().strip().split(',')

    if has_target:

        feature_cols = list(range(df.shape[1] - 1))
        target_col = df.shape[1] - 1

    else:
        feature_cols = list(range(df.shape[1]))
        target_col = None



    if len(feature_cols) != len(saved_features):
        raise ValueError(f"Feature count mismatch: got {len(feature_cols)}, expected {len(saved_features)}")

    X_raw = df[feature_cols].values

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_raw)

    if len(X_scaled) <= seq_length:
        raise ValueError(f"Number of data rows must be greater than {seq_length}")

    X_seq = []
    valid_indices = []
    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_length:i])
        valid_indices.append(i)

    X_seq = np.array(X_seq)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(input_dim=X_raw.shape[1], d_model=64, nhead=4, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_seq).to(device)
    with torch.no_grad():
        pred_log = model(X_tensor).cpu().numpy()

    pred_raw = np.expm1(pred_log)

    df['Predicted_Drilling_Time'] = np.nan
    for idx, pred in zip(valid_indices, pred_raw):
        df.at[idx, 'Predicted_Drilling_Time'] = pred

    if has_target and target_col is not None:
        true_values = df.iloc[valid_indices, target_col].values
        mask = ~np.isnan(true_values) & ~np.isnan(pred_raw)
        if np.sum(mask) > 0:
            mae = mean_absolute_error(true_values[mask], pred_raw[mask])
            r2 = r2_score(true_values[mask], pred_raw[mask])
            print(f"MAE: {mae:.4f}")
            print(f"R^2: {r2:.4f}")

    return df

if __name__ == "__main__":
    mode = 'predict'

    if mode == 'train':
        train_main()

    elif mode == 'predict':

        a = [
            [0.0, 50.0, 60.0, 50.0, 21.0, 1.27, 50.0, 1.0, 32.0, 58.0, 16.9],
            [0.0, 50.0, 60.0, 50.0, 21.0, 1.27, 50.0, 1.0, 32.0, 58.0, 13.0],
            [0.0, 50.0, 60.0, 50.0, 21.0, 1.26, 50.0, 1.0, 32.0, 58.0, 18.9],
            [0.0, 49.0, 60.0, 50.0, 21.0, 1.27, 50.0, 1.0, 32.0, 58.0, 19.0],
            [0.0, 49.0, 60.0, 50.0, 21.0, 1.27, 50.0, 1.0, 32.0, 58.0, 18.1],
            [0.0, 51.0, 60.0, 50.0, 21.0, 1.27, 50.0, 1.0, 32.0, 58.0, 15.2],
            [0.0, 50.0, 60.0, 50.0, 21.0, 1.27, 49.0, 1.0, 32.0, 58.0, 10.3],
            [0.0, 51.0, 60.0, 50.0, 21.0, 1.26, 50.0, 1.0, 32.0, 58.0, 20.3],
            [0.0, 50.0, 60.0, 50.0, 20.0, 1.26, 50.0, 1.0, 32.0, 58.0, 31.0],
            [0.0, 51.0, 60.0, 50.0, 21.0, 1.26, 50.0, 1.0, 32.0, 58.0, 34.4],
            [0.0, 50.0, 60.0, 50.0, 22.0, 1.29, 50.0, 2.0, 64.0, 102.0, 8.6],
            [0.0, 50.0, 60.0, 50.0, 20.0, 1.27, 51.0, 1.0, 32.0, 58.0, 13.8],
            [0.0, 51.0, 60.0, 50.0, 21.0, 1.27, 49.0, 1.0, 32.0, 58.0, 9.9],
            [0.0, 51.0, 60.0, 50.0, 20.0, 1.27, 50.0, 1.0, 32.0, 58.0, 16.0],
            [0.0, 50.0, 60.0, 50.0, 20.0, 1.27, 49.0, 1.0, 32.0, 58.0, 9.1],
            [0.0, 49.0, 60.0, 50.0, 21.0, 1.27, 51.0, 1.0, 32.0, 58.0, 15.7],
            [0.0, 51.0, 60.0, 50.0, 21.0, 1.27, 51.0, 1.0, 32.0, 58.0, 22.1],
            [0.0, 51.0, 60.0, 50.0, 20.0, 1.26, 50.0, 1.0, 32.0, 58.0, 8.5],
            [0.0, 50.0, 60.0, 50.0, 20.0, 1.26, 51.0, 1.0, 32.0, 58.0, 14.9],
            [0.0, 49.0, 60.0, 50.0, 21.0, 1.25, 51.0, 1.0, 32.0, 58.0, 18.0],
            [0.0, 50.0, 60.0, 13.54, 21.0, 1.27, 50.0, 2.06, 110.82, 131.97, 14.24],
            [0.0, 50.0, 60.0, 34.79, 21.0, 1.27, 50.0, 3.26, 95.7, 99.8, 30.27],
            [0.0, 50.0, 60.0, 16.14, 21.0, 1.27, 50.0, 3.17, 40.13, 65.2, 14.01],
            [0.0, 50.0, 60.0, 13.69, 21.0, 1.27, 50.0, 2.87, 57.59, 76.68, 11.81],
            [0.0, 50.0, 60.0, 14.19, 21.0, 1.27, 50.0, 3.01, 36.95, 91.89, 17.33],
            [0.0, 50.0, 60.0, 12.18, 21.0, 1.27, 50.0, 2.66, 126.26, 55.03, 9.43],
            [0.0, 50.0, 60.0, 37.17, 21.0, 1.27, 50.0, 2.32, 26.85, 104.18, 17.82],
            [0.0, 50.0, 60.0, 47.61, 21.0, 1.27, 50.0, 2.27, 51.68, 100.97, 11.06],
            [0.0, 50.0, 60.0, 41.13, 21.0, 1.27, 50.0, 2.25, 54.67, 113.89, 21.11],
            [0.0, 50.0, 60.0, 15.47, 21.0, 1.27, 50.0, 3.43, 39.46, 60.74, 13.88]
        ]

        seq_length = 20
        if len(a) <= seq_length:
            print(f"Error: data row count {len(a)} is less than required {seq_length+1}")
        else:
            result = predict_from_list(a, seq_length=seq_length, has_target=True)
            result.to_excel("predicted_schemes_with_metrics.xlsx", index=False)
    else:
        print("Select 'train' or 'predict'")
