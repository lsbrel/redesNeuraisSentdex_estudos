# IMPLEMENTACAO DE REDE NEURAL EM PYTHON RAW

# CAMADA DE INPUTS
# Nessa camada que vem os dados, sejam eles de fora como no caso de sensores ou algum banco de dados
# e conforme a rede neural se auto-intera eles vem de outras camadas internas ( hidden layers )

#neuron
inputs = [
    1.2,
    2.3,
    4.7
]

#PESOS RELACIONADOS A CADA input de neuron
pesos = [ [0.6,0.18,-0.22], [0.46,0.8,-0.82], [0.3,0.9,-0.72]]


# Vies ou BIAS
#cada neuronio possuem um bias
bias = [1, 4, 7]

# SAIDA

layer_outputs = []
for neuronio_peso, neuronio_bias in zip(pesos, bias):
    saida_neuronio = 0
    for n_input, peso in zip(inputs, neuronio_peso):
        saida_neuronio += n_input*peso
    saida_neuronio += neuronio_bias
    layer_outputs.append(saida_neuronio)

print(layer_outputs)