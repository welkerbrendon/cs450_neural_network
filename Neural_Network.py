from Neuron import Neuron


class Neural_Network:
    neurons = []
    index_neuron_should_have_fired = -1

    def __init__(self, number_layers, number_nodes_per_layer, number_first_layer, number_output):
        self.number_layers = number_layers
        self.number_nodes_per_layer = number_nodes_per_layer
        self.number_first_layer = number_first_layer

        if number_layers == 1:
            for i in range(number_nodes_per_layer[0]):
                neuron = Neuron()
                neuron.is_first_layer = True
                self.neurons.append(neuron)
        else:
            for number in number_nodes_per_layer:
                for i in range(number):
                    neuron = Neuron()
                    if len(self.neurons) < number_first_layer:
                        neuron.is_first_layer = True
                    self.neurons.append(neuron)

        i = 1
        while i <= number_output:
            self.neurons[len(self.neurons) - i].is_output_node = True
            i += 1

    def link_node(self):
        start_indexes, end_indexes = self.get_start_and_end_index_lists()

        i = 0
        j = 1
        while j < len(start_indexes):
            left_layer_current_index = start_indexes[i]
            left_layer_end_index = end_indexes[i]
            right_layer_current_index = start_indexes[j]
            right_layer_end_index = end_indexes[j]
            while right_layer_current_index <= right_layer_end_index:
                while left_layer_current_index <= left_layer_end_index:
                    self.neurons[right_layer_current_index].inputs.append(left_layer_current_index)
                    left_layer_current_index += 1
                right_layer_current_index += 1
                left_layer_current_index = start_indexes[i]
            i += 1
            j += 1

    def get_start_and_end_index_lists(self):
        start_indexes = [0]
        i = 0
        sum_number_0 = 0
        while i < len(self.number_nodes_per_layer) - 1:
            sum_number_0 += self.number_nodes_per_layer[i]
            start_indexes.append(sum_number_0)
            i += 1

        i = 1
        sum_number_1 = self.number_nodes_per_layer[0] - 1
        end_indexes = [sum_number_1]
        while i < len(self.number_nodes_per_layer):
            sum_number_1 += self.number_nodes_per_layer[i]
            end_indexes.append(sum_number_1)
            i += 1

        return start_indexes, end_indexes

    def check_network_configuration(self):
        neuron_counter = 1
        for neuron in self.neurons:
            counter_neurons_as_input = 0
            for input in neuron.inputs:
                if type(input) == Neuron:
                    counter_neurons_as_input += 1
            print("Neuron #" + str(neuron_counter) + " has " + str(counter_neurons_as_input) + " neurons in its input list")
            neuron_counter += 1

    def predict(self, possible_targets, number_output_nodes):
        prediction = None
        output_value = -1

        for target, i in zip(possible_targets, range(1, number_output_nodes)):
            self.neurons[len(self.neurons) - 1].calculate_output
            if self.neurons[len(self.neurons) - i].output > output_value:
                prediction = target
                output_value = self.neurons[len(self.neurons) - i].output
                self.index_neuron_should_have_fired = len(self.neurons) - i

        return prediction

    def calculate_new_weights(self):
        for i in range(len(self.neurons)):
            self.calculate_error(i)

        for neuron in self.neurons:
            for weight in neuron.weights:
                weight = weight - (neuron.error * neuron.output)

    def calculate_error(self, current_index):
        self.neurons[current_index].error = self.neurons[current_index].output * (1 - self.neurons[current_index].output)
        if self.neurons[current_index].is_output_node:
            if current_index == self.index_neuron_should_have_fired:
                self.neurons[current_index].error *= (self.neurons[current_index].output - 1)
            else:
                self.neurons[current_index].error *= self.neurons[current_index].output
        else:
            weight_index_to_get = -1
            start_index_list, end_index_list = self.get_start_and_end_index_lists()
            i = 0
            while weight_index_to_get == -1:
                if current_index >= start_index_list[i] and current_index <= end_index_list[i]:
                    weight_index_to_get = current_index - start_index_list[i]
                    i += 1
                else:
                    i += 1

            summation = 0.0
            for j in range(start_index_list[i], end_index_list[i]):
                summation += self.neurons[j].inputs[weight_index_to_get] * self.neurons[j].error

            self.neurons[current_index].error *= summation

