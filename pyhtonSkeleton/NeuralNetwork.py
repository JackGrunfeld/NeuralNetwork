from math import exp
import numpy as np

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate, biasSwitch):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

        self.biasSwitch = biasSwitch
    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1/ (1+ exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            if self.biasSwitch and i == self.num_hidden - 1:
               hidden_layer_outputs.append(1)
            else:
                weighted_sum = 0.
                for j in range(self.num_inputs):
                  
                    weighted_sum += self.hidden_layer_weights[j][i] * float(inputs[j])
                output = self.sigmoid(weighted_sum)
                hidden_layer_outputs.append(output)
            
        output_layer_outputs = []
        for i in range(self.num_outputs):
           
            weighted_sum = 0.
            for j in range(self.num_hidden):
                weighted_sum += self.output_layer_weights[j][i] * hidden_layer_outputs[j]
          
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)
           
        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs , isPrint):

        output_layer_betas = np.zeros(self.num_outputs)
        output_layer_betas = desired_outputs - output_layer_outputs
        if isPrint:
            print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        for j in range(self.num_hidden):
            for k in range(self.num_outputs):
                hidden_layer_betas[j] += (self.output_layer_weights[j][k] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) *  output_layer_betas[k])
        if isPrint:
            print('HL betas: ', hidden_layer_betas)
        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
              delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) *  output_layer_betas[j]
        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
              delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) *  hidden_layer_betas[j]
        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # TODO! Update the weights.
        self.output_layer_weights += delta_output_layer_weights
        self.hidden_layer_weights += delta_hidden_layer_weights
      

    def train(self, instances, desired_outputs, epochs):
    
        acc  = 0.
 
        for epoch in range(epochs):
            
            print("--------------------------------------------")
            print('epoch = ', epoch)
            print("\n")
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i], False)
                pb = max(output_layer_outputs)
                predicted_class = output_layer_outputs.index(pb)
                predictions.append(predicted_class)

                desired_class = -1
                for t in range(desired_outputs[i].size):
                    if(desired_outputs[i][t] == 1):
                        desired_class = t
                        
                    
                
                if desired_class == predicted_class:
                    acc += 1

                
                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print("\t", 'Hidden layer weights \n', self.hidden_layer_weights)
            print("\t",'Output layer weights  \n', self.output_layer_weights)
          
            # TODO: Print accuracy achieved over this epoch
          
            acc = acc/len(instances)
            print('test acc = ', acc)
            if(acc == 1):
                break
            
            print("\n")

        return acc
    def predict(self, instances):
        predictions = []
        
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            pb = max(output_layer_outputs)
            predicted_class = output_layer_outputs.index(pb)
            predictions.append(predicted_class)
            
        return predictions