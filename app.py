from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F

from model import load_model, CNN1D

app = Flask(__name__, static_folder='static')

# Load your model
model = load_model()

classNames = [
    'Canidae', 'Cervidae', 'CervidaeGazellaSaiga', 'Ovis', 'Equidae',
    'CrocutaPanthera', 'BisonYak', 'Capra', 'Ursidae', 'Vulpes vulpes',
    'Elephantidae', 'Others', 'Rhinocerotidae', 'Rangifer tarandus', 'Hominins'
]


@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json['data']
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1).numpy()[0]

    results = sorted(
        [{'name': classNames[i], 'score': float(probabilities[i])} for i in range(len(classNames))],
        key=lambda x: x['score'], reverse=True
    )
    print(results)
    return jsonify(results)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    app.run(debug=True)
