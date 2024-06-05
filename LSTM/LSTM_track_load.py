import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

model_path = '/content/drive/MyDrive/colab2/AIcup/train_LSTM/weight/last.pth'

input_size = 16
hidden_size = 64
output_size = 1

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

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model if exists
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))

# Move the model to the device
model = model.to(device)

encoder = OneHotEncoder(sparse=False)
# Fit the encoder with all possible 'cam' feature values
encoder.fit(np.array([[0], [1], [2], [3], [4], [5], [6], [7]]))  # 'cam' feature can be 0 to 7


def predict(model, inputs):
    # 將輸入轉換為適當的格式
    inputs = np.array(inputs)
    
    # Extract 'cam' feature and reshape it to 2D array for the encoder
    cam_feature = inputs[0, 0].reshape(-1, 1)
    
    # Transform 'cam' feature to one-hot encoding
    cam_feature_onehot = encoder.transform(cam_feature)
    cam_feature_onehot = np.squeeze(cam_feature_onehot)
    
    # Prepare the sequences
    input_seq = torch.tensor(np.concatenate((cam_feature_onehot, inputs[0, 1:], inputs[1, 1:]), axis=0), dtype=torch.float32)
    
    # 將輸入數據移至適當的設備
    input_seq = input_seq.to(device)
    
    # 使用模型進行預測
    model.eval()
    with torch.no_grad():
        output = model(input_seq.unsqueeze(0).unsqueeze(0))  # Add an extra dimension for batch size
    
    return output.cpu().numpy().squeeze().tolist()

if __name__ == "__main__":
    inputs = [[0, 0, 0, 0, 0],
          [0, 0.40684821605682375, 0.17069200636848572, 0.06779799461364741, 0.09316972702268564]]
    print("只靠上一個軌跡預測",predict(model, inputs))
    inputs = [[0, 0.4717245578765869, 0.19566739400227864, 0.05939416885375977, 0.09901534186469184],
            [0, 0.43120614290237425, 0.27706132464938693, 0.08367326259613037, 0.14628035227457684]]
    print("靠兩個軌跡預測",predict(model, inputs))