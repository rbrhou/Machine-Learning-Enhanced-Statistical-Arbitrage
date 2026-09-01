import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 1: Causal & Temporal Convolutional Network Blocks
# ==============================================================================
class Chomp1d(nn.Module):
    """Slices off trailing right-side padding to enforce strict temporal causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual dilated causal block with weight normalization and dropout."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # 1x1 conv residual projection if channel dims mismatch
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNVaRForecaster(nn.Module):
    """Temporal Convolutional Network projecting sequential features to VaR quantiles."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        quantiles: list[float] = [0.01, 0.05],
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.quantiles = quantiles
        # Fully connected head mapping final temporal embedding to desired quantiles
        self.fc = nn.Linear(num_channels[-1], len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: (batch_size, num_features, seq_len)
        output = self.tcn(x)
        # Extract the representation at the final time step t
        last_step_embedding = output[:, :, -1]
        var_predictions = self.fc(last_step_embedding)
        return var_predictions


# ==============================================================================
# 2: Pinball (Quantile) Loss Function
# ==============================================================================
class PinballLoss(nn.Module):
    """Multi-quantile pinball loss function targeting tail percentiles."""

    def __init__(self, quantiles: list[float] = [0.01, 0.05]):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # y_pred: (batch_size, num_quantiles), y_true: (batch_size, 1)
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, i : i + 1]
            loss_q = torch.max((q - 1) * error, q * error)
            losses.append(loss_q.mean())
        return torch.stack(losses).sum()


# ==============================================================================
# 3: Dataset Pipeline & Sequential 3D Tensor Formatting
# ==============================================================================
class PortfolioSequenceDataset(Dataset):
    """Formats sequential portfolio features into rolling 3D tensors: [batch_size, seq_len, features]."""

    def __init__(
        self, features: np.ndarray, targets: np.ndarray, seq_len: int = 30
    ):
        self.seq_len = seq_len
        self.X, self.y = [], []

        for i in range(len(features) - seq_len):
            self.X.append(features[i : i + seq_len])
            self.y.append(targets[i + seq_len])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).transpose(
            1, 2
        )
        # Transposed to: (N_samples, n_features, seq_len) for Conv1d
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(
            1
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# 4: Verification, Training & Backtesting Suite
# ==============================================================================
def generate_synthetic_portfolio_data(
    n_days: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic heteroskedastic portfolio returns and squared residual features."""
    # GARCH-like dynamic volatility clustering
    vol = np.zeros(n_days)
    returns = np.zeros(n_days)
    vol[0] = 0.01

    for t in range(1, n_days):
        vol[t] = np.sqrt(
            0.00001 + 0.85 * (vol[t - 1] ** 2) + 0.10 * (returns[t - 1] ** 2)
        )
        returns[t] = np.random.normal(0, vol[t])

    # Feature 0: Daily Return; Feature 1: Squared Return (Variance Proxy)
    features = np.column_stack([returns, returns**2])
    return features, returns


def train_and_evaluate_tcn():
    # 1. Prepare Data
    features, returns = generate_synthetic_portfolio_data(n_days=1500)
    split_idx = 1000

    train_dataset = PortfolioSequenceDataset(
        features[:split_idx], returns[:split_idx], seq_len=30
    )
    test_dataset = PortfolioSequenceDataset(
        features[split_idx:], returns[split_idx:], seq_len=30
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Instantiate Model and Loss
    quantiles = [0.01, 0.05]
    model = TCNVaRForecaster(
        num_inputs=2,
        num_channels=[16, 32, 64],
        kernel_size=3,
        dropout=0.1,
        quantiles=quantiles,
    )
    criterion = PinballLoss(quantiles=quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # 3. Training Loop
    model.train()
    for epoch in range(1, 21):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/20 | Pinball Loss: {epoch_loss / len(train_dataset):.6f}"
            )

    # 4. Out-of-Sample VaR Evaluation & Violation Testing
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            all_preds.append(preds)
            all_y.append(batch_y)

    predictions = torch.cat(all_preds, dim=0).numpy()
    actuals = torch.cat(all_y, dim=0).numpy()

    # Calculate empirical violation ratios (Actual Return < -VaR Forecast)
    # Model forecasts negative quantile threshold directly: Return < Predicted Quantile
    v_01 = np.mean(actuals.flatten() < predictions[:, 0])
    v_05 = np.mean(actuals.flatten() < predictions[:, 1])

    print("\n=== Out-of-Sample VaR Diagnostic Results ===")
    print(
        f"1% Target VaR Violation Rate: {v_01:.2%}  (Expected: 1.00% | Exceedance Count: {np.sum(actuals.flatten() < predictions[:, 0])}/{len(actuals)})"
    )
    print(
        f"5% Target VaR Violation Rate: {v_05:.2%}  (Expected: 5.00% | Exceedance Count: {np.sum(actuals.flatten() < predictions[:, 1])}/{len(actuals)})"
    )


if __name__ == "__main__":
    train_and_evaluate_tcn()
