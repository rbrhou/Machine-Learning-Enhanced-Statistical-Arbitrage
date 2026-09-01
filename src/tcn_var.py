import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# 1. Architectural Layers & Causal Convolutions
# ==============================================================================
class Chomp1d(nn.Module):
    """Safely slices trailing right-side padding to enforce strict temporal causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """Residual dilated causal convolutional block with projection alignment."""

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

        # 1x1 projection for residual dimension alignment
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNVaRForecaster(nn.Module):
    """Multi-channel TCN mapping sequential portfolio & macro features to tail quantiles."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int] = [16, 32, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        quantiles: list[float] = [0.01, 0.05],
    ):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
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
        self.fc = nn.Linear(num_channels[-1], len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, num_features, seq_len)
        output = self.tcn(x)
        # Extract representation at final time step t
        last_step = output[:, :, -1]
        return self.fc(last_step)


# ==============================================================================
# 2. Vectorized Quantile (Pinball) Loss
# ==============================================================================
class PinballLoss(nn.Module):
    """Vectorized multi-quantile pinball loss function."""

    def __init__(self, quantiles: list[float] = [0.01, 0.05]):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # y_pred: (batch, n_q), y_true: (batch, 1)
        q = self.quantiles.unsqueeze(0)  # (1, n_q)
        error = y_true - y_pred  # (batch, n_q)
        loss = torch.max(q * error, (q - 1.0) * error)
        return torch.mean(loss)


# ==============================================================================
# 3. Generalized Portfolio Dataset Pipeline
# ==============================================================================
class PortfolioSequenceDataset(Dataset):
    """Constructs rolling temporal 3D tensors: [N_samples, n_features, seq_len]."""

    def __init__(
        self, features: np.ndarray, targets: np.ndarray, seq_len: int = 30
    ):
        self.seq_len = seq_len
        X_list, y_list = [], []

        for i in range(len(features) - seq_len):
            X_list.append(features[i : i + seq_len])
            y_list.append(targets[i + seq_len])

        # Transpose from (N, seq_len, features) to (N, features, seq_len) for Conv1d
        self.X = torch.tensor(np.array(X_list), dtype=torch.float32).transpose(
            1, 2
        )
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(
            1
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ==============================================================================
# 4. Statistical Validation: Kupiec POF Test
# ==============================================================================
def kupiec_pof_test(
    actual_returns: np.ndarray,
    predicted_var: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluates the unconditional coverage of VaR forecasts via the Kupiec POF Likelihood Ratio test."""
    N = len(actual_returns)
    failures = np.sum(actual_returns < predicted_var)
    failure_rate = failures / N if N > 0 else 0.0

    if failures == 0 or failures == N:
        return {
            "alpha": alpha,
            "failures": int(failures),
            "failure_rate": float(failure_rate),
            "lr_stat": 0.0,
            "p_value": 1.0,
            "model_accepted": True,
        }

    # Likelihood Ratio: -2 * ln( ( (1-p)^(N-x) * p^x ) / ( (1 - x/N)^(N-x) * (x/N)^x ) )
    lr_stat = -2.0 * (
        (N - failures) * np.log(1.0 - alpha)
        + failures * np.log(alpha)
        - (N - failures) * np.log(1.0 - failure_rate)
        - failures * np.log(failure_rate)
    )
    p_val = 1.0 - chi2.cdf(lr_stat, df=1)

    return {
        "alpha": alpha,
        "failures": int(failures),
        "failure_rate": float(failure_rate),
        "lr_stat": float(lr_stat),
        "p_value": float(p_val),
        "model_accepted": bool(p_val > 0.05),
    } == "__main__":
    train_and_evaluate_tcn()
    
