import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import random
import mlflow
import mlflow.pytorch
from datetime import date
from mlflow.models import infer_signature

model_path = f"/mnt/proddatalake/dev/TierImminent/model/lstm_{date.today().strftime('%Y_%m_%d')}.pth"

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]
##
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_x)
        out = self.fc(hn[-1])
        #return self.sigmoid(out)
        return out

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.2):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

def train_lstm(sequences, labels, lengths, ids, input_dim, hidden_dim, num_layers, batch_size, learning_rate, num_epochs, balance, logging=False):
    dataset = SequenceDataset(sequences, labels, lengths)

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #############################
    ids_list = ids.tolist() if not isinstance(ids, list) else ids
    indices = list(range(len(ids_list)))
    id_with_indices = list(zip(ids_list, indices))

    random.shuffle(id_with_indices)

    train_split = int(0.8 * len(id_with_indices))
    train_data = id_with_indices[:train_split]
    test_data = id_with_indices[train_split:]

    train_indices = [index for _, index in train_data]
    test_indices = [index for _, index in test_data]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #########################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([balance]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sequences, labels, lengths in train_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            lengths = lengths.cpu()
            optimizer.zero_grad()
            outputs = model(sequences, lengths).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    #torch.save(model, model_path)

    model.eval()
    if logging:
        mlflow.set_experiment(experiment_id="3025905521129250")
        with mlflow.start_run() as run:
            mlflow.log_param("input_size", input_dim)
            mlflow.log_param("hidden_size", hidden_dim)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)

            signature = infer_signature(sequences, labels)
            mlflow.pytorch.log_model(model, f"lstm_pytorch_model_{date.today().strftime('%Y-%m-%d')}", signature=signature)
            run_id = run.info.run_id

    all_labels = []
    all_preds = []

    train_labels = []
    train_preds = []
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            lengths = lengths.cpu()
            outputs = model(sequences, lengths).squeeze()  # Pass lengths to the model
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
        
        for sequences, labels, lengths in train_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            lengths = lengths.cpu()
            outputs = model(sequences, lengths).squeeze()  # Pass lengths to the model
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(outputs.cpu().numpy())

    auc_test = roc_auc_score(all_labels, all_preds)
    print(f"Test AUC: {auc_test:.4f}")

    auc_train = roc_auc_score(train_labels, train_preds)
    print(f"train AUC: {auc_train:.4f}")

    # if logging:
    #     mlflow.log_metric("test_auc", auc_test)
    #     mlflow.log_metric("train_auc", auc_train)

    # fpr, tpr, thresholds = roc_curve(train_labels, train_preds)
    # auc_score = roc_auc_score(train_labels, train_preds)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="best")
    # plt.grid()
    # for i in range(0, len(thresholds), len(thresholds)//20):
    #     plt.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=8, color='red')
    # plt.show()
    # plt.close() 

    return train_labels, train_preds, all_labels, all_preds, run_id

def train_rnn(sequences, labels, lengths):
    input_dim = 18
    hidden_dim = 128
    num_layers = 1
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 10

    dataset = SequenceDataset(sequences, labels, lengths)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNClassifier(input_dim, hidden_dim).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sequences, labels, lengths in train_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths).squeeze()  # Pass lengths to the model
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths).squeeze()  # Pass lengths to the model
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"Test AUC: {auc_score:.4f}")



    