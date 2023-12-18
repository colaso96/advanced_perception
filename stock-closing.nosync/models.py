# Bronte Sihan Li, Cole Crescas Dec 2023
# CS7180

import torch
import torch.nn as nn


class ResCNN(nn.Module):
    """
    Defines a residual CNN architecture for stock prediction.
    This is mostly based on https://github.com/hardyqr/CNN-for-Stock-Market-Prediction-PyTorch/blob/master/source/cnn.py
    """

    def __init__(self, target_series=True):
        super(ResCNN, self).__init__()
        if target_series is True:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.pl = nn.AvgPool2d((5, 2))
            self.fc1 = nn.Linear(2200, 1100)
            self.fc2 = nn.Linear(1100, 200)
            self.tanh = nn.Tanh()
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.3)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 2, kernel_size=1), nn.LeakyReLU(0.3)
            )
            self.pl = nn.AvgPool2d((5, 2))
            self.fc1 = nn.Linear(1364, 682)
            self.fc2 = nn.Linear(682, 1)
            self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer35(out)
        out = self.layer4(out)
        # print(out.size())
        out = self.pl(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.tanh(out)  # TODO: Do we need this?
        return out


class LSTMRegressor(nn.Module):
    """
    LSTMRegressor: PyTorch LSTM-based Neural Network for Regression Tasks

    Parameters:
    - input_size: Number of features in the input.
    - hidden_size: Number of features in the hidden state.
    - num_layers: Number of recurrent layers.
    - output_size: Number of output features.
    - dropout: Dropout probability for the LSTM layer.

    Attributes:
    - lstm: Long Short-Term Memory (LSTM) layer for sequence processing.
    - fc: Fully connected layer for output prediction.
    """

    def __init__(
        self, input_size=200, hidden_size=64, num_layers=2, output_size=1, dropout=0.2
    ):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, input_size)
        x = x.view(x.size(0), -1, x.size(-1))

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Take the output from the last time step
        last_hidden_state = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(last_hidden_state)
        return output


class SimpleTransformer(nn.Module):
    """
    SimpleTransformer: PyTorch Transformer-based Neural Network for Sequence-to-Sequence Tasks

    Parameters:
    - feature_num: Number of input features.
    - d_model: Dimensionality of the model's hidden states.
    - nhead: Number of heads in the multiheadattention models.
    - num_layers: Number of sub-encoder-layers in the Transformer.

    Attributes:
    - embedding: Linear layer for feature dimension reduction.
    - tf1: First Transformer layer for encoding.
    - fc: Fully connected layer with dropout for additional feature processing.
    - dropout: Dropout layer to prevent overfitting.
    - tf2: Second Transformer layer for encoding.
    - decoder: Linear layer for output prediction.
    """

    def __init__(
        self, feature_num=200, d_model=96, nhead=8, num_layers=1, dropout_rate=0.25
    ):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(feature_num, d_model)
        self.tf1 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.tf2 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(d_model, 200)

    def forward(self, x):
        # Reduce the feature dimension
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.embedding(x)
        x = self.tf1.encoder(x)
        x = self.fc(x[:, -1, :])  # Use the last sequence output
        x = self.dropout(x)
        x = self.tf2.encoder(x)
        x = self.decoder(x)
        return x


class TimeSeriesTransformer(nn.Module):
    """
    TimeSeriesTransformer: PyTorch Transformer-based Neural Network for Time Series Forecasting.  This did not work well
    since the pooling action can inhibit learning from surrounding information.

    Parameters:
    - feature_num: Number of input features.
    - d_model: Dimensionality of the model's hidden states.
    - nhead: Number of heads in the multiheadattention models.
    - num_layers: Number of sub-encoder-layers in the Transformer.
    - dropout_rate: Dropout rate for regularization.

    Attributes:
    - embedding: Linear layer for feature dimension reduction.
    - transformer1: First Transformer layer for encoding temporal patterns.
    - attention_pooling: Attention pooling layer to capture relevant temporal information.
    - fc: Fully connected layer with dropout for additional feature processing.
    - transformer2: Second Transformer layer for encoding temporal dependencies.
    - decoder: Linear layer for output prediction.
    """

    def __init__(
        self, feature_num=200, d_model=96, nhead=8, num_layers=1, dropout_rate=0.25
    ):
        super(TimeSeriesTransformer, self).__init__()

        # Input feature dimension reduction
        self.embedding = nn.Linear(feature_num, d_model)

        # First Transformer layer for encoding temporal patterns
        self.transformer1 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )

        # Attention pooling to capture relevant temporal information
        self.attention_pooling = nn.MultiheadAttention(d_model, nhead)

        # Fully connected layer with dropout
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Second Transformer layer for encoding temporal dependencies
        self.transformer2 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )

        # Output prediction layer
        self.decoder = nn.Linear(d_model, 200)

    def forward(self, x):
        # Reduce the feature dimension
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.embedding(x)

        # First Transformer layer for encoding temporal patterns
        x = self.transformer1.encoder(x)

        # Attention pooling to capture relevant temporal information
        x, _ = self.attention_pooling(x, x, x)

        # Fully connected layer with dropout
        x = self.fc(x[:, -1, :])  # Use the last sequence output
        x = self.dropout(x)

        # Second Transformer layer for encoding temporal dependencies
        x = self.transformer2.encoder(x)

        # Output prediction
        x = self.decoder(x).squeeze(dim=-1)  # Adjust for regression
        return x


class ThreeLayerTransformer(nn.Module):
    """
    ThreeLayerTransformer: PyTorch Transformer-based Neural Network for Sequence-to-Sequence Tasks adding another transformer layer

    Parameters:
    - feature_num: Number of input features.
    - d_model: Dimensionality of the model's hidden states.
    - nhead: Number of heads in the multiheadattention models.
    - num_layers: Number of sub-encoder-layers in the Transformer.

    Attributes:
    - embedding: Linear layer for feature dimension reduction.
    - tf1: First Transformer layer for encoding.
    - fc: Fully connected layer with dropout for additional feature processing.
    - dropout: Dropout layer to prevent overfitting.
    - tf2: Second Transformer layer for encoding.
    - tf3: Third Transformer layer for encoding.
    - decoder: Linear layer for output prediction.
    """

    def __init__(
        self, feature_num=200, d_model=96, nhead=8, num_layers=1, dropout_rate=0.25
    ):
        super(ThreeLayerTransformer, self).__init__()
        self.embedding = nn.Linear(feature_num, d_model)
        self.tf1 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        # Second Transformer layer
        self.tf2 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        # Third Transformer layer
        self.transformer3 = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(d_model, 200)

    def forward(self, x):
        # Reduce the feature dimension
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.embedding(x)
        x = self.tf1.encoder(x)
        x = self.fc(x[:, -1, :])  # Use the last sequence output
        x = self.dropout(x)
        x = self.tf2.encoder(x)
        x = self.decoder(x)
        return x
