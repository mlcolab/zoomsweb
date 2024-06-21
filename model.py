import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InferenceModel:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.class_names = [
            'Canidae', 'Cervidae', 'CervidaeGazellaSaiga', 'Ovis', 'Equidae',
            'CrocutaPanthera', 'BisonYak', 'Capra', 'Ursidae', 'Vulpes vulpes',
            'Elephantidae', 'Others', 'Rhinocerotidae', 'Rangifer tarandus', 'Hominins'
        ]

        self.ref_peaks = {
            name: np.loadtxt(f'reference_data/{name}.csv').tolist() for name in self.class_names
        }

    def __call__(self, data):
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1).numpy()[0]

        results = sorted(
            [
                {
                    'name': name,
                    'score': 100 * round(float(probabilities[i]), 3),
                    'peaks': self.ref_peaks[name],
                } for i, name in enumerate(self.class_names)],
            key=lambda x: x['score'], reverse=True
        )
        # only send results with score > 0.01
        results = [r for r in results if r['score'] >= 1.]

        for i, result in enumerate(results):
            result['id'] = i

        return results


class CNN1D(nn.Module):
    def __init__(self, input_size: int = 5201, num_classes: int = 15):
        super().__init__()

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
    model = CNN1D()
    model.load_state_dict(torch.load(f'models/{name}.pth', map_location='cpu'))
    return InferenceModel(model)
