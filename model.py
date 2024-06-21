import numpy as np
import onnxruntime as ort


class InferenceModel:
    def __init__(self, name: str):
        self.model = ort.InferenceSession(f'models/{name}.onnx')

        self.class_names = [
            'Canidae', 'Cervidae', 'CervidaeGazellaSaiga', 'Ovis', 'Equidae',
            'CrocutaPanthera', 'BisonYak', 'Capra', 'Ursidae', 'Vulpes vulpes',
            'Elephantidae', 'Others', 'Rhinocerotidae', 'Rangifer tarandus', 'Hominins'
        ]

        self.ref_peaks = {
            name: np.loadtxt(f'reference_data/{name}.csv').tolist() for name in self.class_names
        }

    def __call__(self, data):
        data = np.array(data).reshape(1, 1, -1).astype(np.float32)
        output = self.model.run(["output"], {"input": data})[0].reshape(-1)
        probabilities = softmax(output)

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


def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def load_model(name: str = 'model'):
    return InferenceModel(name)
