import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for LSTMModel."""
    def __init__(self, input_dim=10, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2, 
                 lr=0.001, batch_size=32, epochs=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.criterion = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)
            
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            y_tensor = y.unsqueeze(1).to(self.device)

        # Reshape X for LSTM: (batch_size, seq_len, input_dim) -> Here we assume feature vector
        # If input is (N, Features), we treat it as seq_len=1 for simplicity in Voting,
        # OR we need to reshape upstream.
        # Standard tabular data doesn't have seq_len.
        # LSTM is for time-series.
        # If using VotingClassifier on the SAME dataset as XGBoost (tabular), LSTM needs sequence provided.
        # For this refactor, we will assume input is (N, InputDim) and unsqueeze to (N, 1, InputDim).
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1) # (N, 1, Features)

        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            # Simple full batch training for prototype (implement mini-batch for real prod)
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)
            
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)
            
        self.model.eval()
        with torch.no_grad():
            prob = self.model(X_tensor).cpu().numpy()
        return np.hstack([1-prob, prob])

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_len, hidden_dim)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
