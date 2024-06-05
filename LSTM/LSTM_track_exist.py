"""
讀取exist.npy裡的資料來訓練，
每個資料內含有3個array，array[0]到array[2]皆表示['cam', 'x', 'y', 'width', 'height']
其中 cam 是需要做one-hot-encoding的類別，代表攝影機編號，x, y 代表物件的中心座標，width, height 代表物件的寬和高
其中模型輸入array[0]和array[1]作為sequence，然後輸出0或1判斷物體是否還在0為不在，1為還在，
而array[2]則是label，其後四項的總和(即sum(array[2][1:]))若為0表示物體不在，若不為0表示物體還在
請幫我先處裡好label，將其轉為0或1。

幫我將以項程式改為此二元分類題目

我想讓20%的資料作為驗證集，80%的資料作為訓練集

請更改以下程式碼，使其符合上述需求
"""
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from torchinfo import summary
from torchviz import make_dot
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


epochs = 100
input_size = 12
hidden_size = 64
output_size = 1
lr = 0.001
batch_size = 16

encoder = OneHotEncoder(sparse_output=False)

# Load data from .npy file
data = np.load('/content/drive/MyDrive/colab2/AIcup/32_33_train_v2/exist.npy', allow_pickle=True)

# Extract 'cam' feature and reshape it to 2D array for the encoder
cam_feature = np.array([track[0][0] for track in data]).reshape(-1, 1)

# Fit the encoder and transform 'cam' feature to one-hot encoding
cam_feature_onehot = encoder.fit_transform(cam_feature)

# Prepare the sequences
input_seq = [
    [torch.tensor(np.concatenate((cam_feature_onehot[i], track[j, 1:]), axis=0), dtype=torch.float32)
     for j in range(2)]
    for i, track in enumerate(data)]
labels = [torch.tensor(1 if np.sum(track[2, 1:]) > 0 else 0, dtype=torch.float32) for track in data]  # Change labels to binary
# print(input_seq[0])
# Split data into training and validation sets
input_seq_train, input_seq_val, labels_train, labels_val = train_test_split(input_seq, labels, test_size=0.2, random_state=42)
print(len(labels_train), len(labels_val))

# Convert lists to tensors
input_seq_train = [torch.stack(seq) for seq in input_seq_train]
input_seq_val = [torch.stack(seq) for seq in input_seq_val]
input_seq_train = pad_sequence(input_seq_train, batch_first=True)
labels_train = torch.stack(labels_train).squeeze()  # Remove the extra dimension
input_seq_val = pad_sequence(input_seq_val, batch_first=True)
labels_val = torch.stack(labels_val).squeeze()  # Remove the extra dimension

# Create dataloaders
train_data = TensorDataset(input_seq_train, labels_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data = TensorDataset(input_seq_val, labels_val)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)  # Add 1D convolutional layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)  # Add layer normalization

    def forward(self, x):
        x = x.transpose(1, 2)  # Swap the dimensions for 1D convolution
        x = self.conv1d(x)  # Apply 1D convolution
        x = x.transpose(1, 2)  # Swap the dimensions back
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.ln(out)  # Apply layer normalization
        out = self.fc(out[:, -1, :])
        return out.squeeze()  # Squeeze the output to remove the last dimension

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Print model summary
print(summary(model, input_size=(1, 10, input_size)))

# Plot model architecture
make_dot(
    model(torch.randn(1, 1, input_size).to(device)),
    params=dict(model.named_parameters())
).render("/content/model", format="png")

# Training loop
train_epoch_losses = []
val_epoch_losses = []
train_acc_scores = []
val_acc_scores = []
all_preds = []
all_labels = []

for epoch in range(epochs):
    model.train()
    train_losses = []
    train_corrects = 0
    total_train = 0
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        output = model(seq)
        # print(output, label)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        preds = torch.round(torch.sigmoid(output))  # Apply sigmoid and round to get binary predictions
        train_corrects += (preds == label).sum().item()
        total_train += label.size(0)

    train_acc = train_corrects / total_train
    train_acc_scores.append(train_acc)

    model.eval()
    val_losses = []
    val_corrects = 0
    total_val = 0
    all_preds = []  # Reset the lists at the start of each epoch
    all_labels = []
    with torch.no_grad():
        for seq, label in val_loader:
            seq, label = seq.to(device), label.to(device)  # Move data to the same device as the model
            output = model(seq)
            loss = criterion(output, label)
            val_losses.append(loss.item())
            preds = torch.round(torch.sigmoid(output))  # Apply sigmoid and round to get binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            val_corrects += (preds == label).sum().item()
            total_val += label.size(0)

    val_acc = val_corrects / total_val
    val_acc_scores.append(val_acc)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_losses):.3f}, Validation Loss: {np.mean(val_losses):.3f}, Train Acc: {train_acc:.3f}, Validation Acc: {val_acc:.3f}')

    # Add the average losses for this epoch to the lists
    train_epoch_losses.append(np.mean(train_losses))
    val_epoch_losses.append(np.mean(val_losses))

# Print model summary
print(summary(model, input_size=(1, 10, input_size)))

# train_precision = precision_score(label.cpu().numpy(), preds.cpu().numpy())
# train_recall = recall_score(label.cpu().numpy(), preds.cpu().numpy())
# train_f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy())
# train_confusion = confusion_matrix(label.cpu().numpy(), preds.cpu().numpy())

val_precision = precision_score(all_labels, all_preds)
val_recall = recall_score(all_labels, all_preds)
val_f1 = f1_score(all_labels, all_preds)
val_confusion = confusion_matrix(all_labels, all_preds)

# 建立混淆矩陣的視覺化
# train_disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion)
val_disp = ConfusionMatrixDisplay(confusion_matrix=val_confusion)

# 繪製混淆矩陣
# train_disp.plot()
val_disp.plot()

# print(f'Train Precision: {train_precision:.3f}, Train Recall: {train_recall:.3f}, Train F1: {train_f1:.3f}')
# print(f'Train Confusion Matrix:\n {train_confusion}')
print(f'Validation Precision: {val_precision:.3f}, Validation Recall: {val_recall:.3f}, Validation F1: {val_f1:.3f}')
# print(f'Validation Confusion Matrix:\n {val_confusion}')

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_scores, label='Training Accuracy')
plt.plot(val_acc_scores, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(train_epoch_losses, label='Training Loss')
plt.plot(val_epoch_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# 假設 `model` 是你的模型
# torch.save(model.state_dict(), '/content/drive/MyDrive/colab2/AIcup/train_LSTM/weight/exist_last.pth')


# Load model if exists
model_path = '/content/drive/MyDrive/colab2/AIcup/train_LSTM/weight/exist_last.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))

