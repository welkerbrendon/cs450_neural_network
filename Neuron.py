from random import uniform

class Neuron:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.weights = list()
        self.bias = -1
        self.bias_weight = uniform(-1.0, 1.0)

    def set_weights(self):
        end = len(self.inputs[0])
        i = 0
        while i < end:
            self.weights.append(uniform(-1.0, 1.0))
            i += 1


    def calculate_weighted_input(self, row):
        weighted_input = self.bias * self.bias_weight
        for input, weight in zip(row, self.weights):
            weighted_input += (input * weight)
        return weighted_input

    def calculate_outputs(self):
        for row in self.inputs:
            weighted_input = self.calculate_weighted_input(row)
            if weighted_input > 0:
                self.outputs.append(1)
            else:
                self.outputs.append(0)
