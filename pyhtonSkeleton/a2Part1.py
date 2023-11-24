import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import Neural_Network


BiasSwitch = True

def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


if __name__ == '__main__':
    
    data = pd.read_csv('penguins307-train.csv')
    pd_data_ts = pd.read_csv('penguins307-test.csv')
    # the class label is last!
    labels = data.iloc[:, -1]
    test_labels = pd_data_ts.iloc[:, -1]
    
    # seperate the data from the labels    
    test_instances = pd_data_ts.iloc[:, :-1]
    instances = data.iloc[:, :-1]
    if BiasSwitch:
    # add a column of 1s to the data to represent the bias
        instances["bias"] = 1 
        test_instances["bias"] = 1

    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    test_instances = scaler.fit_transform(test_instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.1

   
    if BiasSwitch:
        
        initial_hidden_layer_weights = np.array([[-0.28, -0.22, 0], [0.08, 0.20, 0], [-0.30, 0.32,0], [0.10, 0.01,0], [-0.02, -0.2,0]])
        initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36], [-0.33, 0.26, 0.06]])

        print("h: " , initial_hidden_layer_weights)
        print("o: " , initial_output_layer_weights)
        n_in += 1
        n_hidden += 1
    else:
        initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
        initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    
    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate, BiasSwitch)
    
    print('\n First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))
    print("output: ", nn.forward_pass(instances[0]))
    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))
    
    #print the updated weights of the network
    print("hidden layer weights: ", nn.hidden_layer_weights) 
    print("output layer weights: ", nn.output_layer_weights)
    
    
    
    

    # TODO: Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    weights = nn.output_layer_weights
    ops = nn.forward_pass(instances[0])
    deltas = nn.backward_propagate_error(instances[0], ops[0], ops[1], onehot_encoded[0], True)
    nn.update_weights(deltas[0], deltas[1])
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    # TODO: Train for 100 epochs, on all instances.
    num_epochs = 600
    
    acc = nn.train(instances,onehot_encoded , num_epochs)
    
    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # TODO: Compute and print the test accuracy
    correct = 0
    predLabels = nn.predict(test_instances)
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(test_labels)
    
    for i in range(len(predLabels)):
        if(predLabels[i] == integer_encoded[i]):
            correct += 1
            
    test_accuracy = correct / len(test_instances)
    print("--------------------------------------------")
    print('\nTest accuracy: ', round(test_accuracy * 100,2), '%')