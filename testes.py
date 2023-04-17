import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

entradas = [
            [1,2,3],
            [2.3,2,2.4]
            ]

class Camada:
    # Funcaod e init que cria a camada, recebe o numero de inputs e o numero de neuronios na camada
    def __init__(self, num_inputs, num_neuronios):
        # pesos recebe valores aleatorios gerados para uma matriz de num_inputs em linhas e num_neuronios
        # em colunas, tudo isso vezes 0.10 para que seja um valor entre -1 e 1
        self.pesos = 0.01*np.random.randn(num_inputs, num_neuronios)
        # Biases recebe uma matriz de 1 por numero de neuronios de zeros
        self.biases = np.zeros((1, num_neuronios))

    # Parte da execucao da funcao
    def forward(self, valores_entrada):
        # resultado_camada recebe o a soma de produtos dos inputs e pesos e depois o bias é adicionado
        self.resultado_camada = np.dot(valores_entrada, self.pesos) + self.biases
    

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



class Activation_softmax:
    def forward(self, inputs):
        # Get unormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probs
    



X, y = spiral_data(samples=10, classes=2)


primeiraCamada = Camada(2,3)

ativacao1 = Activation_ReLU()

segundaCamada = Camada(3,3)

ativacao2 = Activation_softmax()




primeiraCamada.forward(X)
ativacao1.forward(primeiraCamada.resultado_camada)
segundaCamada.forward(ativacao1.output)
ativacao2.forward(segundaCamada.resultado_camada)

print(f'{ativacao2.output}')

# print(f'{X}')
# print('----------')
# print(f'{primeiraCamada.pesos}')
# print('----------')
# print(f'{primeiraCamada.resultado_camada}')
