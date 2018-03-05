from random import uniform
import math

class Neuron:
    def __init__(self):
        self.inputs = []
        self.weighted_inputs = []
        self.output = 0.0
        self.weights = list()
        self.bias = -1
        self.bias_weight = uniform(-1.0, 1.0)
        self.error = 0.0
        self.is_first_layer = False
        self.is_output_node = False

    def set_weights(self):
        end = len(self.inputs[0])
        i = 0
        while i < end:
            self.weights.append(uniform(-1.0, 1.0))
            i += 1

    def calculate_weighted_input(self, neuron_input):
        if neuron_input:
            for input, weight in zip(self.inputs, self.weights):
                self.weighted_inputs.append(input.output * weight)
        else:
            for input, weight in zip(self.inputs, self.weights):
                self.weighted_inputs.append(input * weight)

    def calculate_output(self):
        summation = 0.0
        if type(self.inputs[0]) is Neuron:
            for neuron in self.inputs:
                neuron.calculate_output()

            self.calculate_weighted_input(True)
        else:
            self.calculate_weighted_input(False)
            summation = sum(self.weighted_inputs)

        self.output = 1 / (1 + math.exp(-summation))

