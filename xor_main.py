from NeuralNetwork import *
import random

train_data = [
    {
        "inputs": [0, 0],
        "output": [0]
    },
    {
        "inputs": [1, 0],
        "output": [1]
    },
    {
        "inputs": [0, 1],
        "output": [1]
    },
    {
        "inputs": [1, 1],
        "output": [0]
    }
]


nn = NeuralNetwork(2, 3, 1, 0.3)

print(nn.query([0, 0]))
print(nn.query([1, 0]))
print(nn.query([0, 1]))
print(nn.query([1, 1]))

for i in range(50000):
    data = train_data[random.randint(0, 3)]
    nn.train(data["inputs"], data["output"])

print("------------------")
print(nn.query([0, 0]))
print(nn.query([1, 0]))
print(nn.query([0, 1]))
print(nn.query([1, 1]))


