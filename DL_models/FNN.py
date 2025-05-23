import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

def extract_features(train_df, test_df, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_df['preprocessed_sentence'])
    X_test = vectorizer.transform(test_df['preprocessed_sentence'])
    return X_train, X_test, vectorizer

def prepare_dataloaders(X_train, y_train, X_val, y_val, batch_size=64, device='cpu'):
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    return train_loader, val_loader

def compute_class_weight(y_train):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return torch.tensor([pos_weight], dtype=torch.float32)

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            all_preds.extend((preds > 0.5).int().tolist())
            all_labels.extend(yb.int().tolist())

    print(classification_report(all_labels, all_preds, target_names=["Not Summary", "Summary"]))


def save_model(model, path="fnn_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path, input_size):
    model = FeedForwardNet(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
