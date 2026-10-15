"""
Model Architectures and Estimator Factory.
Strictly implements the 10 classifiers and Table 4 hyperparameters from the research paper:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Decision Tree
  4. Random Forest
  5. XGBoost
  6. AdaBoost
  7. Multilayer Perceptron (MLP)
  8. 1D Convolutional Neural Network (CNN)
  9. Long Short-Term Memory (LSTM)
  10. Gated Recurrent Unit (GRU)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import SEED, EPOCHS_MLP, EPOCHS_DL


# ---------------------------------------------------------------------------
# PyTorch Architectures (CNN, LSTM, GRU)
# ---------------------------------------------------------------------------

class PyTorchBaseClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for PyTorch tabular/sequence classifiers."""
    
    def __init__(self, model_type="CNN", epochs=EPOCHS_DL, batch_size=64, lr=0.0005, seed=SEED):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.classes_ = np.array([0, 1])
        self.model_ = None

    def _build_network(self, input_dim):
        import torch
        import torch.nn as nn
        
        torch.manual_seed(self.seed)
        
        if self.model_type == "CNN":
            # Paper Table 4: Conv1D (64, 128 filters, kernel size 3, ReLU)
            class CNNNet(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                    self.relu1 = nn.ReLU()
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.relu2 = nn.ReLU()
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.fc = nn.Linear(128, 2)
                    
                def forward(self, x):
                    # x shape: (batch, 1, features)
                    x = self.relu1(self.conv1(x))
                    x = self.relu2(self.conv2(x))
                    x = self.pool(x).squeeze(-1)
                    return self.fc(x)
                    
            return CNNNet(input_dim)

        elif self.model_type == "LSTM":
            # Paper Table 4: LSTM (100 units, Tanh, Dropout 0.3)
            class LSTMNet(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=100, batch_first=True, dropout=0.3)
                    self.fc = nn.Linear(100, 2)
                    
                def forward(self, x):
                    # x shape: (batch, features, 1)
                    out, (hn, _) = self.lstm(x)
                    return self.fc(hn[-1])
                    
            return LSTMNet(input_dim)

        elif self.model_type == "GRU":
            # Paper Table 4: GRU (80 units, Tanh, Dropout 0.2)
            class GRUNet(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.gru = nn.GRU(input_size=1, hidden_size=80, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(80, 2)
                    
                def forward(self, x):
                    # x shape: (batch, features, 1)
                    out, hn = self.gru(x)
                    return self.fc(hn[-1])
                    
            return GRUNet(input_dim)

        else:
            raise ValueError(f"Unknown deep learning model_type: {self.model_type}")

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        
        n_samples, n_features = X_arr.shape
        self.model_ = self._build_network(n_features)
        
        # Prepare inputs based on architecture:
        # CNN expects (batch, channels=1, seq_len=features)
        # LSTM/GRU expects (batch, seq_len=features, input_size=1)
        if self.model_type == "CNN":
            X_tensor = torch.tensor(X_arr).unsqueeze(1)
        else:
            X_tensor = torch.tensor(X_arr).unsqueeze(-1)
            
        y_tensor = torch.tensor(y_arr)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        
        if self.model_type == "GRU":
            # Paper Table 4: RMSprop(lr=0.001) for GRU
            optimizer = torch.optim.RMSprop(self.model_.parameters(), lr=self.lr)
        else:
            # Paper Table 4: Adam(lr=0.0005) for CNN and LSTM
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        if self.model_type == "CNN":
            X_tensor = torch.tensor(X_arr).unsqueeze(1)
        else:
            X_tensor = torch.tensor(X_arr).unsqueeze(-1)
            
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ---------------------------------------------------------------------------
# Estimator Factory Matching Paper Table 4
# ---------------------------------------------------------------------------

def create_model(model_name, seed=SEED):
    """
    Instantiate model with exact hyperparameters from Table 4 of the research paper:
      - Logistic Regression: Solver liblinear, penalty l2
      - SVM: Kernel RBF
      - Decision Tree: Max depth None
      - Random Forest: Number of estimators 100
      - XGBoost: lr 0.1, max depth 6, n_estimators 100
      - AdaBoost: n_estimators 50
      - MLP: Hidden layers [128, 64, 32], ReLU, Adam lr 0.001, batch 32, epochs 80
      - CNN: Filters 64, 128, kernel 3, ReLU, Adam lr 0.0005, batch 64, epochs 40
      - LSTM: Units 100, Tanh, Dropout 0.3, Adam lr 0.0005, batch 64, epochs 60
      - GRU: Units 80, Tanh, Dropout 0.2, RMSprop lr 0.001, batch 128, epochs 50
    """
    clean_name = model_name.strip()
    
    if clean_name in ["Logistic Regression", "LogReg", "lr"]:
        return LogisticRegression(
            solver="liblinear",
            penalty="l2",
            random_state=seed,
            max_iter=1000
        )
        
    elif clean_name in ["SVM", "Support Vector Machine", "svm"]:
        return SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=seed
        )
        
    elif clean_name in ["Decision Tree", "DT", "dt"]:
        return DecisionTreeClassifier(
            max_depth=None,
            criterion="gini",
            random_state=seed
        )
        
    elif clean_name in ["Random Forest", "RandomForest", "rf"]:
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            random_state=seed,
            n_jobs=-1
        )
        
    elif clean_name in ["XGBoost", "xgboost", "xgb"]:
        return XGBClassifier(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )
        
    elif clean_name in ["AdaBoost", "adaboost", "ada"]:
        return AdaBoostClassifier(
            n_estimators=50,
            random_state=seed
        )
        
    elif clean_name in ["MLP", "Multilayer Perceptron", "mlp"]:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=32,
            max_iter=EPOCHS_MLP,
            early_stopping=True,
            random_state=seed
        )
        
    elif clean_name in ["CNN", "Convolutional Neural Network", "cnn"]:
        return PyTorchBaseClassifier(
            model_type="CNN",
            epochs=min(40, EPOCHS_DL),
            batch_size=64,
            lr=0.0005,
            seed=seed
        )
        
    elif clean_name in ["LSTM", "lstm"]:
        return PyTorchBaseClassifier(
            model_type="LSTM",
            epochs=min(60, EPOCHS_DL),
            batch_size=64,
            lr=0.0005,
            seed=seed
        )
        
    elif clean_name in ["GRU", "gru"]:
        return PyTorchBaseClassifier(
            model_type="GRU",
            epochs=min(50, EPOCHS_DL),
            batch_size=128,
            lr=0.001,
            seed=seed
        )
        
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from Table 4 models.")
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from Table 4 models.")
