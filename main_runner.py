from Neural_Network import Neural_Network
from sklearn import datasets
from scipy import stats
import pandas as pd
import numpy as np

number_neurons = -1
while number_neurons <= 0:
    number_neurons = int(input("Please enter the number of neurons you would like to run with: "))
    if number_neurons <= 0:
        print("Please enter a number")

neuron_list = Neural_Network(number_neurons).get_neuron_list()

print("Number of neurons = ", len(neuron_list))

correct_input = False
targets = None
data = None
while not correct_input:
    which_dataset = input("Please enter the dataset you would like to process (Irirs = I, Pima Diabetes = P): ")
    if 'I' in which_dataset:
        data = datasets.load_iris().data
        targets = datasets.load_iris().target
        correct_input = True
    elif 'P' in which_dataset:
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data").values.tolist()
        targets = np.array(dataset.pop())
        data = np.array(dataset)
        correct_input = True
    else:
        print("Incorrect input, please try again choosing either Iris or Pima datasets.")

z_score_normailized_data = stats.zscore(data)
length_of_data_rows = len(z_score_normailized_data[0])
index_jump_by = int(np.math.ceil(length_of_data_rows / float(number_neurons)))
if index_jump_by > 1:
    end_index_for_input = index_jump_by
else:
    end_index_for_input = 2
beginning_index_for_input = 0

count = 1

for neuron in neuron_list:
    if count < number_neurons:
        neuron.inputs = z_score_normailized_data[:, beginning_index_for_input:end_index_for_input]
        beginning_index_for_input = end_index_for_input
        end_index_for_input += index_jump_by
        count += 1
    else:
        neuron.inputs = z_score_normailized_data[:, beginning_index_for_input:]

for neuron in neuron_list:
    neuron.set_weights()

for neuron in neuron_list:
    print("Weights: ", neuron.weights)

for neuron in neuron_list:
    neuron.calculate_outputs()

neuron_number = 0
for neuron in neuron_list:
    neuron_number += 1
    for output in neuron.outputs:
        print("Outputs for Neuron #", neuron_number, ": ", output)