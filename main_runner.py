from Neural_Network import Neural_Network
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy import stats

import pandas as pd
import numpy as np

number_layers = -1
number_output_nodes = -1
while number_layers <= 0:
    number_layers = int(input("Please enter the number of layers you would like to run with: "))
    if number_layers <= 0:
        print("Please enter a number greater than 0")
number_hidden_nodes = []
temp_counter = 1
while len(number_hidden_nodes) < number_layers - 1:
    input_value = -1
    while input_value <= 0 :
        input_request_string = "Please enter the number of hidden nodes you would like to run with in layer " + str(temp_counter) + ": "
        input_value = int(input(input_request_string))
        if input_value <= 0:
            print("Please enter a number greater than 0")
    number_hidden_nodes.append(input_value)
    temp_counter += 1
while number_output_nodes <= 0:
    number_output_nodes = int(input("Please enter the number of output nodes you would like to run with: "))
    if number_output_nodes <= 0:
        print("Please enter a number greater than 0")
number_hidden_nodes.append(number_output_nodes)
network = Neural_Network(number_layers, number_hidden_nodes, number_hidden_nodes[0], number_output_nodes)
network.link_node()
network.check_network_configuration()

print("Number of neurons = ", len(network.neurons))
first_layer_node_count = 0
for neuron in network.neurons:
    if neuron.is_first_layer:
        first_layer_node_count += 1
print("Number in first layer = ", first_layer_node_count)

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
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data").values
        targets = dataset[:,len(dataset[0]) - 1]
        data = dataset[:,0:len(dataset[0]) - 2]
        correct_input = True
    else:
        print("Incorrect input, please try again choosing either Iris or Pima datasets.")

data_train, data_test, target_train, target_test = train_test_split(data, targets, test_size=.3)

z_score_normailized_data = stats.zscore(data_train)
length_of_data_rows = len(z_score_normailized_data[0])

inputs_per_node = length_of_data_rows / first_layer_node_count
number_nodes_to_get_extra_input = length_of_data_rows % first_layer_node_count

correctly_predicted_count = 0
for row, target in zip(z_score_normailized_data, target_train):
    for data in row:
        i = 0
        while i < first_layer_node_count:
            network.neurons[i].inputs.append(data)
            i += 1
    i = 1
    while i <= number_output_nodes:
        network.neurons[len(network.neurons) - i].calculate_output
        i += 1

    target_set = set(targets)
    prediction = network.predict(list(target_set), number_output_nodes)

    if prediction != target:
        network.calculate_new_weights()




