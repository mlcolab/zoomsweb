import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from model import load_model, CNN1D

app = Flask(__name__, static_folder='static')

# Load your model
model = load_model()


@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json['data']
    results = model(data)
    return jsonify(results)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    app.run(debug=True)
