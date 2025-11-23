import torch
import torch.nn as nn

class ConvLSTMClassifier(nn.Module):
    """
    Input:
      seq: (batch, T, 3)
      feat: (batch, F)
    Architecture:
      - Conv1D layers across time on the 3-channel input (T,3 -> conv expects (batch, channels, T))
      - Permute conv outputs to (batch, T', channels') then LSTM (batch, T', hidden)
      - Global max pool on LSTM outputs or use last hidden
      - Concatenate engineered features -> FC layers -> logits
    """
    def __init__(self, seq_channels=3, feat_dim=30, lstm_hidden=128, lstm_layers=1, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # after pooling, time dimension reduced; but LSTM returns 2*hidden dims for bidirectional
        self.fc = nn.Sequential(
            nn.Linear(2*lstm_hidden + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, seq, feat):
        # seq: (batch, T, 3)
        x = seq.permute(0,2,1)  # -> (batch, channels=3, T)
        x = self.conv(x)        # -> (batch, channels=64, T')
        x = x.permute(0,2,1)    # -> (batch, T', channels)
        out, (hn, cn) = self.lstm(x)  # out:(batch, T', 2*hidden)
        # use mean pooling over time dimension
        out_pool = out.mean(dim=1)    # (batch, 2*hidden)
        # concat engineered features
        combined = torch.cat([out_pool, feat], dim=1)
        logits = self.fc(combined)
        return logits
