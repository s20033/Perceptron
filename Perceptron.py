import numpy as np

import pandas as pd


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]  # dot product matrx multiplication bias add
        if sum > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[0] += self.learning_rate * (label - prediction)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs

                #leru type of sigmoid function.
                #to predict either 1/0.



# read and  modify data
data = pd.read_csv('perceptron.data')
data = data[:]
data[4] = np.where(data.iloc[:, -1] == 'Iris-setosa', 0, 1)
training_inputs = data.iloc[:, 0:4].values
labels = data.iloc[:, -1].values
# activate the perceptron
perceptron = Perceptron(4)
# train the perceptron
perceptron.train(training_inputs, labels)

# test the perceptron
test_data = pd.read_csv('perceptron.test.data')

test_data = test_data[:]
test_data[4] = np.where(test_data.iloc[:, -1] == 'Iris-setosa', 0, 1)

test_data_input = test_data.iloc[:, 0:4].values

# count number of total missclassified
miss_classified = 0
miss_classified_list = []
test_data.iat[0, 5]
for i in range(0, test_data_input.shape[0]):
    test_prediction1 = perceptron.predict(test_data_input[i])
    miss_classified_list.append(test_prediction1)
    if test_prediction1 != test_data.iat[i, 5]:
        miss_classified = miss_classified + 1
accuracy = (1 - (miss_classified / test_data.shape[0])) * 100
print("Accuracy: ", accuracy)  # Accuracy is here giving 100%, with the test data as provided in Assignment.


# for Experimental purpose, I change the test data to check whether accuracy changes if test data is changed.
# After making some random changes in Test data the Accuracy is decreased making the code quite reliable.

# input the data in array

# I have put some random input manually(not UI input data) for checking the implementation of my code.
# It seems the code is working well. however the Accuracy is 100% which is pretty weird.

# input = [4.6, 6.2, 1.4, 0.2]
# output_prediction = perceptron.predict(input)
# if output_prediction == 0:
#   print('iris sitosa')
# elif output_prediction == 1:
#   print('iris-versicolor')

# take a input from user (UI input of Data)
list_input = [0, 0, 0, 0]
for i in range(0, 4):
    input_user = float(input('give a  input of i th value :   '))
    list_input[i] = input_user
print('the data you have Provided is : ', list_input)

output_prediction = perceptron.predict(list_input)
if output_prediction == 0:
    print('According to the input you have given, : iris sitosa')
elif output_prediction == 1:
    print('According to the input you have given : iris-versicolor')
