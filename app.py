from flask import Flask, request, jsonify, send_from_directory

from model import load_model

app = Flask(__name__, static_folder='static')
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


def main():
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


if __name__ == '__main__':
    main()
