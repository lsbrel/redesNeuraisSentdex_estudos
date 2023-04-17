import numpy as np

# np.random.seed(0)

X = [
        [1, 2, 4, 2.5],
        [2.0, 5.1, -1.0, 2],
        [-1.5, 2.7, 2.2, -3]
    ] 


class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weigths = 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weigths) + self.biases

    def showBiases(self):
        return self.biases
    
    def showWeights(self):
        return self.weigths

teste = Layer_dense(4, 3)
teste.forward(X)


print(f'Biases => {teste.showBiases()}')
print(f'Weights => {teste.showWeights()}')
print(f'Output => {teste.output}')