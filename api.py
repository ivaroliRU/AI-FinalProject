from flask import Flask, request, jsonify
from network import NeuralNetwork

network = NeuralNetwork(filename='network.npz')
app = Flask(__name__)

def normalize(arr):
    newArr = []

    for item in arr:
        newArr.append(item/10)
    return newArr


@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.get_json(silent=True)

    if('input' in body and len(body['input']) == 9):
        try:
            input = normalize(body['input'])
            prediction = network.run(input).tolist()
            
            response = {"malignant": prediction[1], "benign": prediction[0]}

            return jsonify(response)
        except:
            return jsonify({"Reason":"Incorrectly formatted request"}), 400

    return jsonify({"Reason":"Incorrectly formatted request"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)