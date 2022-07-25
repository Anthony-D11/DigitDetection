from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy import sum, matmul, transpose, log, zeros
from scipy.optimize import fmin_tnc
from sklearn.linear_model import LogisticRegression
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
def create_dense(layer_sizes):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation='sigmoid', input_shape=(image_size,)))
    for s in layer_sizes[1:]:
        model.add(Dense(s, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def evaluate(model, batch_size=128, num_epochs=5):
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=.1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
image_size = 784 # 28*28
num_classes = 10 # ten unique digits

X_train = X_train.reshape(X_train.shape[0], image_size)
X_test = X_test.reshape(X_test.shape[0], image_size)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

for layers in range(1, 5):
    model = create_dense([32]*layers)
    evaluate(model, num_epochs=40)

#Train using Logistic Regression
'''
model = CustomLogisticRegression(learning_rate, num_epochs, _lambda)
all_theta = []
cost_list = []
for k in range(K):
    label = [1 if y_train[i] == k else 0 for i in range(y_train.shape[0])]
    label = np.matrix(label)
    label = np.reshape(label, (label.shape[1], 1))
    cost, theta = model.fit(X_train, label)
    all_theta.append(theta)
    cost_list.append(cost)
    print(cost)
    print(f'Done for class {k}')

pred = model.predict(X_train, all_theta)

plotCost(num_epochs, cost_list[0])

c = 0
for i in range(y_train.shape[0]):
    if pred[i] == y_train[i]:
        c += 1

print('accuracy -> {}'.format(c / y_train.shape[0]))

size = 5
count = 0
for i in range(size*size):
    plt.subplot(size, size, 1+i)
    plt.imshow(X_test[i], cmap=plt.get_cmap('gray'))
    test = np.reshape(X_test[i], (1, X_test.shape[1] * X_test.shape[2]))

    p = model.predict(test, all_theta)

    if i % size == 0:
        print('\n')
    print(p, end=' ')

    if p == y_test[i]:
        count += 1
    plt.title(model.predict(test, all_theta))
plt.show()
print(f'right: {count}')


with open('all_theta.csv', 'w') as outputFile:
    for k in range(K):
        outputFile.write(str(all_theta[k][0]))
        for i in range(len(all_theta[k])):
            if i > 0:
                outputFile.write(',')
                outputFile.write(str(all_theta[k].item(i)))
    outputFile.close()

'''

#Train using neural network


