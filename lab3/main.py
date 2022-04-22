import os
import random
from PIL import Image
from scipy.special import expit
import cupy as np

expit = np.ElementwiseKernel('float64 x', 'float64 y', 'y = 1 / (1 + exp(-x))', 'expit')


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_grade):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_grade = learning_grade

        # weights from input to hidden layer
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        # Sigmoid
        self.activation_function = lambda x: expit(x)

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learning_grade * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                 np.transpose(hidden_outputs))

        self.wih += self.learning_grade * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                 np.transpose(inputs))

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def save(self):
        with open('who.txt', 'w') as who_file:
            for row in self.who:
                r = []
                for el in row:
                    r.append(str(el))
                who_file.write(f"{';'.join(list(r))}\n")

        with open('wih.txt', 'w') as wih_file:
            for row in self.wih:
                r = []
                for el in row:
                    r.append(str(el))
                wih_file.write(f"{';'.join(list(r))}\n")

    def load(self):
        with open('who.txt', 'r') as who_file:
            who = who_file.readlines()
            who = [r.strip() for r in who]
            who = [r.split(';') for r in who]
            who = [[float(el) for el in r] for r in who]
            self.who = np.asarray(who)
        with open('wih.txt', 'r') as wih_file:
            wih = wih_file.readlines()
            wih = [r.strip() for r in wih]
            wih = [r.split(';') for r in wih]
            wih = [[float(el) for el in r] for r in wih]
            self.wih = np.asarray(wih)


def get_image_data(image_name):
    image = Image.open(image_name)
    data = np.array(image)
    data = 255 - data
    data = data / 255.0
    data = data * 0.99
    data = data + 0.01
    data = data.flatten()
    return data


def load_dataset(dataset_name):
    pictures = []
    dir_name = f'dataset/{dataset_name}'
    for filename in os.listdir(dir_name):
        picture_name = f'{dir_name}/{filename}'
        pictures.append(get_image_data(picture_name))
    return pictures


def train(datasets):
    input_nodes = 2500
    hidden_nodes = 150
    output_nodes = len(datasets)
    learning_grade = 0.2
    training_epochs = 175
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_grade)
    for e in range(training_epochs):
        print(f'Training epoch {e}')
        for i, d in enumerate(datasets):
            targets = np.zeros(output_nodes) + 0.01
            targets[i] = 0.99
            pictures = load_dataset(d)
            for picture in pictures:
                n.train(picture, targets)

    n.save()


if __name__ == "__main__":

    datasets = ['add', 'dec', 'div', 'eq', 'mul', 'sub']
    # train(datasets)

    input_nodes = 2500
    hidden_nodes = 150
    output_nodes = len(datasets)
    learning_grade = 0.2
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_grade)
    n.load()

    for d in datasets:
        img_path = f'tests/{d}.jpg'
        img_data = get_image_data(img_path)
        p = n.query(img_data).flatten()
        index_max = int(np.argmax(p))
        print(f'Classified element {img_path} of dataset {d} as {datasets[index_max]}')
