import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.pool = nn.AvgPool1d(kernel_size=3)

        output_size = (input_size - 5 + 1) // 3
        output_size = (output_size - 5 + 1) // 3

        self.fc1 = nn.Linear(64 * output_size, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


def load_model(name: str = 'model'):
    model = torch.load(f'models/{name}.pth', map_location='cpu')
    model.eval()
    return model
