import numpy as np
from Neuron import Neuron


class Neural_Network:
    neurons = []

    def __init__(self, number_neurons):
        while number_neurons > 0:
            self.neurons.append(Neuron())
            number_neurons -= 1

    def get_neuron_list(self):
        return np.array(self.neurons)