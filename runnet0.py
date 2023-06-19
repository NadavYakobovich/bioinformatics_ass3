import sys

import numpy as np

from buildnet0 import NeuralNetwork

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split()
            string = line[0]
            data.append(string)
    return data

def load_weights_from_file(file_name):
    file = open(file_name, "r")
    # read the first line and get the size of the network
    line = file.readline()
    layer1_size , layer2_size = line.split()
    layer1_size = int(layer1_size)
    layer2_size = int(layer2_size)
    weights1 = []
    weights2 = []
    for i in range(layer1_size):
        line = file.readline()
        weights1.append([float(x) for x in line.split()])
    for i in range(layer2_size):
        line = file.readline()
        weights2.append([float(x) for x in line.split()])
    file.close()
    best_individual = NeuralNetwork()
    best_individual.weights1 = np.array(weights1)
    best_individual.weights2 = np.array(weights2)
    return best_individual



#this function get nural network and data and return a txt file that in every line there is the prediction of the network on the data
def write_predictions_to_file(network, data):
    file = open("output0.txt", "w")
    correct_predictions_counter = 0
    for input_data in data:
        prediction = network.forward(input_data)
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == 1:
            file.write(str('1') + "\n")
        else:
            file.write(str('0') + "\n")
    file.close()

def preprocess_data(data):
    processed_data = []
    for string in  data:
        # Convert string to numpy array of integers
        encoded_string = np.array([int(char) for char in string])
        processed_data.append(encoded_string)
    return processed_data

#get a file that in each line there is a string and a label and write 2 file one with the string an one with the label
def create2file(file):
    file1 = open("test0.txt", "w")
    file2 = open("label0.txt", "w")
    with open(file, 'r') as file:
        for line in file:
            line = line.split()
            string = line[0]
            label = line[1]
            file1.write(string + "\n")
            file2.write(label + "\n")
    file1.close()
    file2.close()

def comparePrediction(outputfile, labelFile):
    count = 0
    output = open(outputfile, "r")
    label = open(labelFile, "r")
    size = 0
    for line1, line2 in zip(output, label):
        size += 1
        if line1 == line2:
            count += 1
    #print the accuracy
    print("accuracy: " + str(count/size * 100) + "%")

def main(wnet, data ):
    network = load_weights_from_file(wnet)
    test_data = load_data(data)
    test_data = preprocess_data(test_data) #make the data to be in the right format (np array)
    write_predictions_to_file(network, test_data)

if __name__ == '__main__':
    # get from args 2 arrays the first is the name of the file and the second is the data
    wnet = sys.argv[1] # the Wnet file
    data = sys.argv[2] # the data to test
    main(wnet, data)
    # comparePrediction("output0.txt", "label0.txt") # if you want to check the accuracy of the network
