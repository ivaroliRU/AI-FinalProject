import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from dataparser import getNormalizedData

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
    def __init__(self, 
                 no_of_in_nodes = 0, 
                 no_of_out_nodes = 0, 
                 no_of_hidden1_nodes = 0,
                 no_of_hidden2_nodes = 0,
                 learning_rate = 0,
                 filename=""):
        if filename == "":
            self.no_of_in_nodes = no_of_in_nodes
            self.no_of_out_nodes = no_of_out_nodes
            self.no_of_hidden1_nodes = no_of_hidden1_nodes
            self.no_of_hidden2_nodes = no_of_hidden2_nodes
            self.learning_rate = learning_rate 
            self.create_weight_matrices()
        else:
            print("1")
            self.load_network(filename)
        
    def create_weight_matrices(self):
        #Setting up the weights for input into the fist hidden layer
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden1 = X.rvs((self.no_of_hidden1_nodes, 
                                       self.no_of_in_nodes))
        
        #Setting up the weights from the first hidden layer into the next
        rad = 1 / np.sqrt(self.no_of_hidden1_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden2 = X.rvs((self.no_of_hidden2_nodes, 
                                        self.no_of_hidden1_nodes))
        
        #Setting up the weights from the last hidden layer into the output
        rad = 1 / np.sqrt(self.no_of_hidden2_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden2_nodes))
        
    
    def train(self, input_vector, target_vector):
        #Take in input and target vector
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        #Calculate the value vector for the first hidden layer
        output_vector1 = np.dot(self.weights_in_hidden1, input_vector)
        output_vector_hidden1 = activation_function(output_vector1)
        
        #calculate the value vector for the second hidden layer
        output_vector2 = np.dot(self.weights_in_hidden2, output_vector_hidden1)
        output_vector_hidden2 = activation_function(output_vector2)

        #calculate the value vector for the output layer
        output_vector3 = np.dot(self.weights_hidden_out, output_vector_hidden2)
        output_vector_network = activation_function(output_vector3)
        
        #calculate output errors
        output_errors = target_vector - output_vector_network
        
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden2.T)
        self.weights_hidden_out += tmp


        # calculate seccond hidden layer errors:
        hidden2_errors = np.dot(self.weights_hidden_out.T, output_errors)

        # update the weights:
        tmp = hidden2_errors * output_vector_hidden2 * (1.0 - output_vector_hidden2)
        self.weights_in_hidden2 += self.learning_rate * np.dot(tmp, output_vector_hidden1.T)


        # calculate first hidden layer errors:
        hidden1_errors = np.dot(self.weights_in_hidden2.T, hidden2_errors)

        # update the weights:
        tmp = hidden1_errors * output_vector_hidden1 * (1.0 - output_vector_hidden1)
        self.weights_in_hidden1 += self.learning_rate * np.dot(tmp, input_vector.T)
           
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_in_hidden1, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.weights_in_hidden2, output_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector

    def save_network(self, filename):
        #save weight arrays into a file
        np.savez(filename, layer1= self.weights_in_hidden1, 
                                layer2= self.weights_in_hidden2, 
                                layer3=self.weights_hidden_out)
    
    def load_network(self, filename):
        #load weight data
        data = np.load(filename)
        self.weights_in_hidden1 = data['layer1']
        self.weights_in_hidden2 = data['layer2']
        self.weights_hidden_out = data['layer3']


if __name__ == "__main__":
    network = NeuralNetwork(no_of_in_nodes=9, 
                               no_of_out_nodes=2, 
                               no_of_hidden1_nodes=7,
                               no_of_hidden2_nodes=5,
                               learning_rate=0.6,
                               filename="")
    #network = NeuralNetwork(filename='network.npz')
    trainData = getNormalizedData('data/TrainData.csv')
    testData = getNormalizedData('data/TestData.csv')
    
    num = 1
    incorrect, correct, samples = 0,0,0

    while num < 1000:
        for row in testData:
            calculated = network.run(row['input'])
            
            if(calculated[0] > calculated[1]):
                if(row['output'][0] == 1):
                    correct = correct + 1
            else:
               if(row['output'][1] == 1):
                    correct = correct + 1
            samples = samples + 1      
        print(str(num) + "," + str((correct/samples)))
        #print("Accuracy: " + str((correct/samples)*100) + "%")

        for index, row in enumerate(trainData):
            network.train(row['input'], row['output'])
        num = num + 1

    #network.save_network("network")