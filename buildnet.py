from statistics import mean
from graph import PercentPerGeneration
import sys


# get from the txt for each line a string and the answer
def getData(txt_name):
    data_list = []
    answer_list = []
    with open(txt_name, 'r') as file:
        for line in file:
            # Split the line into words
            words = line.split()
            string_bit = words[0]
            answer = words[1]
            data_list.append(string_bit)
            answer_list.append(answer)
    return data_list, answer_list


import numpy as np
import random

# Genetic Algorithm parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 350
MUTATION_RATE = 0.01
TRAIN_RATIO = 0.6
RANDOM_RATIO = 0.1
TRAIN_DATA_PATH = 'train.txt'


# Neural Network parameters
INPUT_SIZE = 16
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1


# Load training data from file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split()
            string = line[0]
            label = int(line[1])
            data.append((string, label))
    return data

def split_train_test_data(data, train_ratio=TRAIN_RATIO):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Preprocess the data
def preprocess_data(data):
    processed_data = []
    for string, label in data:
        # Convert string to numpy array of integers
        encoded_string = np.array([int(char) for char in string])
        processed_data.append((encoded_string, label))
    return processed_data


# Define the neural network architecture
class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.randn(INPUT_SIZE * HIDDEN_SIZE).reshape(INPUT_SIZE, HIDDEN_SIZE)   - 0.5 # 16 * 32
        self.weights2 = np.random.randn(HIDDEN_SIZE * OUTPUT_SIZE).reshape(HIDDEN_SIZE, OUTPUT_SIZE)  - 0.5  # 32 * 1

    def forward(self, x):
        self.hidden = np.dot(x, self.weights1)
        self.hidden_activation = self.sigmoid(self.hidden)
        self.output = np.dot(self.hidden_activation, self.weights2)
        self.output_activation = self.sigmoid(self.output)
        return self.output_activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def copy(self):
        copy = NeuralNetwork()
        copy.weights1 = self.weights1.copy()
        copy.weights2 = self.weights2.copy()
        return copy


# Genetic Algorithm
def genetic_algorithm(data):
    population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
    global RANDOM_RATIO
    best, worst, avrg = [], [], []

    for generation in range(NUM_GENERATIONS):
        print("Generation {}".format(generation + 1))
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness_scores.append(evaluate_fitness(individual, data))

        # Create new generation through crossover and mutation
        offspring = new_population(population, fitness_scores)
        worst.append(min(fitness_scores) * 100)
        avrg.append(mean(fitness_scores) * 100)
        best.append(max(fitness_scores) * 100)
        #print the score of the best individual in the generation
        print("Best individual score: {:.2f}%".format(max(fitness_scores) * 100))

        population = offspring

    # Select the fittest individual
    best_individual = max(population, key=lambda x: evaluate_fitness(x, data))
    PercentPerGeneration(best, worst, avrg)
    return best_individual
#get from the population the top 5 individuals with the highest fitness score
def new_population(population, fitness_scores):
    new_population_list = []
    #sort the population by the fitness score reverse
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
    sorted_population.reverse()
    new_population_list.append(sorted_population[0].copy())
    # add the 60 copy of the top netWork, each with muted version
    for i in range(60):
        muted_individual = mutate(sorted_population[0].copy())
        new_population_list.append(muted_individual)

    #get the top 20% of the population
    top_population = sorted_population[:int(len(sorted_population)*0.20)]
    for _ in range(POPULATION_SIZE - len(new_population_list)):
        parent1, parent2 = random.choice(population), random.choice(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population_list.append(child)

    return new_population_list





# Evaluate fitness based on neural network performance
def evaluate_fitness(individual, data):
    correct_predictions = 0
    for input_data, label in data:
        prediction = individual.forward(input_data)
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == label:
            correct_predictions += 1
    fitness = correct_predictions / len(data)
    return fitness


# Select parents for crossover based on fitness scores
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    parents = np.random.choice(population, size=POPULATION_SIZE, p=probabilities)
    return parents


# Perform crossover between two parents
def crossover(parent1, parent2):
    child = NeuralNetwork()
    child.weights1 = (parent1.weights1 + parent2.weights1) / 2
    child.weights2 = (parent1.weights2 + parent2.weights2) / 2
    return child


# Perform mutation on an individual
# def mutate(individual):
#     for weight in [individual.weights1, individual.weights2]:
#         mask = np.random.uniform(0, 1, size=weight.shape) < MUTATION_RATE
#         random_values = np.random.randn(*weight.shape)
#         weight[mask] += random_values[mask]
#     return individual

def mutate(individual):
    for weight in [individual.weights1, individual.weights2]:
        mask = np.random.uniform(0, 1, size=weight.shape) < MUTATION_RATE
        random_values = np.random.choice([-RANDOM_RATIO, RANDOM_RATIO], size=weight.shape)
        weight[mask] += random_values[mask]
    return individual


#this function get the best individual and write its weights to a file
def write_weights_to_file(best_individual):
    file = open("wnet.txt", "w")
    #write the size of the layers
    file.write( str(INPUT_SIZE) + " " + str(HIDDEN_SIZE) + "\n")
    for row in best_individual.weights1:
        for weight in row:
            file.write(str(weight) + " ")
        file.write("\n")
    for row in best_individual.weights2:
        for weight in row:
            file.write(str(weight) + " ")
        file.write("\n")
    file.close()

# Main code
def main():
    # Load and preprocess data
    data = load_data('nn1.txt')
    train_data , test_data = split_train_test_data(data)
    processed_data = preprocess_data(train_data)

    # Run genetic algorithm to evolve the neural network
    best_individual = genetic_algorithm(processed_data)

    # Test the best individual on new data
    processed_test_data = preprocess_data(test_data)

    correct_predictions = 0
    for input_data, label in processed_test_data:
        prediction = best_individual.forward(input_data)
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(processed_test_data)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    write_weights_to_file(best_individual)


if __name__ == '__main__':
    data_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    main()


