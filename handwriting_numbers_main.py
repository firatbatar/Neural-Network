from NeuralNetwork import *
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
import imageio

np.set_printoptions(threshold=np.inf)


def convert_image_to_list(img_file_dir):
    img_array = np.zeros(28 * 28)
    img_array_temp = imageio.imread(img_file_dir)
    index = 0
    for i in range(len(img_array_temp)):
        for j in range(len(img_array_temp[i])):
            img_array[index] = img_array_temp[i, j, 0]
            index += 1

    img_data = 255.0 - img_array
    img_data = (img_data / 255.0 * 0.99) + 0.01
    return img_data


# def string_to_ndarray(lines):
#     for line in lines:
#         line = line.strip()
#         line = line.replace("\n", "")
#         line = line.replace("[", "")
#         line = line.replace("]", "")
#         line.strip()
#         elements = line.split(" ")
#         while "" in elements:
#             elements.remove("")
#         elements = list(map(float, elements))
#         new_list.append(elements)
#     new_a = np.array(new_list)
#     return new_a


# Load training and test datasets.
with open("mnist_datasets/mnist_train_60000.csv", "r") as file:
    train_data = []
    for i in range(60000):
        train_data.append({})
        values = file.readline().split(',')

        train_data[i]["input"] = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

        target = np.zeros(10) + 0.01
        target[int(values[0])] = 0.99
        train_data[i]["target"] = target

with open("mnist_datasets/mnist_test_10000.csv", "r") as file:
    test_data = []
    for i in range(10000):
        test_data.append({})
        values = file.readline().split(',')

        test_data[i]["input"] = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

        target = np.zeros(10) + 0.01
        target[int(values[0])] = 0.99
        test_data[i]["target"] = target


# Create the neural network with 784 inputs (one for each pixel) and 10 outputs (one for each digit)
nn = NeuralNetwork(784, 100, 10, 0.1)

start = time()

# Train the network 5 times
for epoch in range(5):
    for data in train_data:
        nn.train(data["input"], data["target"])

print("Training is completed!")
print(f"Training time: {time() - start}")

# # Save the current state of the neural network into a txt file
# nn.save_state("nn_save")

# Test the network
correct = 0
wrong = 0

for data in test_data:
    target = np.where(data["target"] == 0.99)[0][0]

    q = nn.query(data["input"])
    guess = 0
    for i in range(10):
        if q[i][0] > q[guess][0]:
            guess = i

    if guess == target:
        correct += 1
    else:
        wrong += 1


# Print the accuracy
print(f"\nCorrect: {correct}\nWrong: {wrong}")
print(f"Accuracy: {correct/(correct + wrong)}")

print(f"\nTotal  time: {time() - start}")
