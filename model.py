import torch
import torch.nn as NeuralN

#neural network model used in chat.py
class NeuralNet(NeuralN.Module):
    #method to take input data and process in 3 defined layers
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #NeuralN.Linear() function Applies a linear transformation to the incoming data: :math:y = xA^T + b
        self.layer1 = NeuralN.Linear(input_size, hidden_size) 
        self.layer2 = NeuralN.Linear(hidden_size, hidden_size) 
        self.layer3 = NeuralN.Linear(hidden_size, num_classes)
        #Applies the rectified linear unit function element-wise, works as a activation function for the layers 
        self.relu = NeuralN.ReLU()
    
    #giving the forward pass in data layers
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        #no need of activation of reLU as this is the last layer
        return out

