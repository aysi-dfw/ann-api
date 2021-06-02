import flask
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
K = tf.keras

class NeuralNet:
    def __init__(self):
        self.net = None
        self.X, self.Y = self.get_data()
        self.scaler = StandardScaler().fit(self.X)

    def predict(self, ar):
        assert len(ar) == 3
        probs = self.net.predict(self.scaler.transform(np.array(ar)[None, :]))[0]
        amax = np.argmax(probs)
        return int(amax), float(probs[amax])

    def train_model(self):
        net = K.Sequential([
            K.layers.Dense(100, input_shape=(3,), activation='relu'),
            K.layers.Dense(100, activation='relu'),
            K.layers.Dense(2, activation='softmax')
        ])
        net.compile(
            optimizer=K.optimizers.Adam(learning_rate=0.001),
            loss=K.losses.SparseCategoricalCrossentropy(),
            metrics=[K.metrics.SparseCategoricalAccuracy()]
        )
        net.fit(self.scaler.transform(self.X), self.Y, validation_split=0.05, epochs=500)

        return net
    
    @staticmethod
    def get_data():
        X, Y = [], []
        with open('haberman.data', 'r') as f:
            for line in f:
                cur = list(map(int, line.strip(' \t\r\n').split(',')))
                X.append(cur[:3])
                Y.append(cur[3] - 1)
        X, Y = np.array(X), np.array(Y)
        return X, Y

M = NeuralNet()
M.net = M.train_model()

app = flask.Flask(__name__)


@app.route('/api/nn_predict')
def predict():
    if 'data' not in flask.request.args:
        return 'ERROR: Please send data', 400
    data = flask.request.args['data']
    return flask.jsonify(M.predict(list(map(int, data.split(',')))))


if __name__ == '__main__':
    app.run()