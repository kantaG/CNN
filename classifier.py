import layers

class CNN:
    def __init__(self, input_layer_size, hidden_layers, output_layer_size):
        layer_list = []
        layer_sizes = [input_layer_size]
        
        for i in range(len(hidden_layers)):
            print(layer_sizes)
            if hidden_layers[i][0] == "conv":
                layer_list.append(layers.Convolution(layer_sizes[i], hidden_layers[i][1]))
                layer_sizes.append([hidden_layers[i][1], (layer_sizes[i][1] - 3) + 1, (layer_sizes[i][2] - 3) + 1])
            elif hidden_layers[i][0] == "relu":
                layer_list.append(layers.ReLU)
                layer_sizes.append(layer_sizes[i])
            elif hidden_layers[i][0] == "pool":
                layer_list.append(layers.Pooling(layer_sizes[i]))
                layer_sizes.append([layer_sizes[i][0], int(1 + (layer_sizes[i][1] - 2) / 2), int(1 + (layer_sizes[i][2] - 2) / 2)])
            elif hidden_layers[i][0] == "flat":
                layer_list.append(layers.Flatten)
                layer_sizes.append([1, layer_sizes[i][0] * layer_sizes[i][1] * layer_sizes[i][2]])
            elif hidden_layers[i][0] == "line":
                layer_list.append(layers.Linear(layer_sizes[i], hidden_layers[i][1]))
                layer_sizes.append([1, hidden_layers[i][1]])
            
        layer_list.append(layers.Output(layer_sizes[len(layer_sizes)-1], output_layer_size))
        layer_sizes.append([1, output_layer_size])
        
    # def predict(self):
        
    # def forward(self):
        
    # def evaluate_accuracy(self):