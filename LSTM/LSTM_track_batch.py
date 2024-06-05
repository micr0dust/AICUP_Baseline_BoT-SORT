"""
讀取data.npy裡的資料來訓練，其中data.npy有10038個資料，
每個資料內含有3個array，array[0]到array[2]皆表示['cam', 'x', 'y', 'width', 'height']
其中 cam 是需要做one-hot-encoding的類別，代表攝影機編號，x, y 代表物件的中心座標，width, height 代表物件的寬和高
其中模型輸入array[0]和array[1]作為sequence，然後輸出['x', 'y', 'width', 'height']，
而array[2]則是label，
預測要跟array[2]的['x', 'y', 'width', 'height']算loss。
array[2]的'cam'不會用到

每個資料皆獨立紀載一個物體的3步軌跡，用前兩步預測第三步
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

epochs = 100
input_size = 12
hidden_size = 64
output_size = 4
lr = 0.001
batch_size = 16

encoder = OneHotEncoder(sparse_output=False)

# Load data from .npy file
data = np.load('/content/drive/MyDrive/colab2/AIcup/32_33_train_v2/data.npy', allow_pickle=True)

# Extract 'cam' feature and reshape it to 2D array for the encoder
cam_feature = np.array([track[0][0] for track in data]).reshape(-1, 1)

# Fit the encoder and transform 'cam' feature to one-hot encoding
cam_feature_onehot = encoder.fit_transform(cam_feature)

# Prepare the sequences
input_seq = [
    [torch.tensor(np.concatenate((cam_feature_onehot[i], track[j, 1:]), axis=0), dtype=torch.float32)
     for j in range(2)]
    for i, track in enumerate(data)]
labels = [torch.tensor(track[2, 1:], dtype=torch.float32) for track in data]
print(input_seq[0])
# Split data into training and validation sets
input_seq_train, input_seq_val, labels_train, labels_val = train_test_split(input_seq, labels, test_size=0.2, random_state=42)

# Convert lists to tensors
input_seq_train = [torch.stack(seq) for seq in input_seq_train]
input_seq_val = [torch.stack(seq) for seq in input_seq_val]
input_seq_train = pad_sequence(input_seq_train, batch_first=True)
labels_train = torch.stack(labels_train)
input_seq_val = pad_sequence(input_seq_val, batch_first=True)
labels_val = torch.stack(labels_val)

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
        return out

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Print model summary
print(summary(model, input_size=(1, 10, input_size)))

# Plot model architecture
make_dot(
    model(torch.randn(1, 1, input_size).to(device)),
    params=dict(model.named_parameters())
).render("/content/model", format="png")

def bbox_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = np.maximum(boxA[:, 0], boxB[:, 0])
    yA = np.maximum(boxA[:, 1], boxB[:, 1])
    xB = np.minimum(boxA[:, 2], boxB[:, 2])
    yB = np.minimum(boxA[:, 3], boxB[:, 3])

    # Compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou

# Training loop
train_epoch_losses = []
val_epoch_losses = []
train_iou_scores = []
val_iou_scores = []

for epoch in range(epochs):
    model.train()
    train_losses = []
    train_ious = []
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_ious.append(bbox_iou(output.detach().cpu().numpy(), label.cpu().numpy()).mean())  # Move data back to CPU for numpy operations

    model.eval()
    val_losses = []
    val_ious = []
    with torch.no_grad():
        for seq, label in val_loader:
            seq, label = seq.to(device), label.to(device)  # Move data to the same device as the model
            output = model(seq)
            loss = criterion(output, label)
            val_losses.append(loss.item())
            val_ious.append(bbox_iou(output.detach().cpu().numpy(), label.cpu().numpy()).mean())  # Move data back to CPU for numpy operations
    print(f'Train Loss: {np.mean(train_losses):.3f}, Validation Loss: {np.mean(val_losses):.3f}, Train IOU: {np.mean(train_ious):.3f}, Validation IOU: {np.mean(val_ious):.3f}')

    # Add the average losses and IOU score for this epoch to the lists
    train_epoch_losses.append(np.mean(train_losses))
    val_epoch_losses.append(np.mean(val_losses))
    train_iou_scores.append(np.mean(train_ious))
    val_iou_scores.append(np.mean(val_ious))

# 假設 `model` 是你的模型
# torch.save(model.state_dict(), '/content/drive/MyDrive/colab2/AIcup/train_LSTM/weight/last.pth')

# Plot training and validation IOU
plt.plot(train_iou_scores, label='Training IOU')
plt.plot(val_iou_scores, label='Validation IOU')
plt.legend()
plt.show()

# Load model if exists
model_path = '/content/drive/MyDrive/colab2/AIcup/train_LSTM/weight/last.pth'
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path, map_location=device))
