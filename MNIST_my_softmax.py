import os
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("FMNIST_data/", one_hot=True)

def softmax(x):
    exp_minmax = lambda x: np.exp(x-np.max(x))
    denom = lambda x: 1.0/np.sum(x)
    x = np.apply_along_axis(exp_minmax,1,x)
    denominator = np.apply_along_axis(denom,1,x)
    if len(denominator.shape) == 1:
        denominator = denominator.reshape((denominator.shape[0],1))

    x = x*denominator
    return x

def predict(w, X):
    return softmax(np.dot(X, w))

def accuracy(y_, Y):
    max_index = np.argmax(y_, axis = 1)
    y_[np.arange(y_.shape[0]), max_index] = 1
    accuracy = np.sum(np.argmax(y_,axis=1)==np.argmax(Y,axis=1))
    accuracy = accuracy*1.0/Y.shape[0]
    return accuracy

def propagation(w, c, X, Y):
    m = X.shape[0]
    A = softmax(np.dot(X,w))
    J = -1 / m * np.sum(Y*np.log(A))
    # J += 0.5*c*np.sum(w**2)
    dw = -1/m * np.dot(X.T, (Y-A))
    # dw += c*w
    
    update = {'dw' : dw, 'cost' : J}
    return update

def optimization(learning_rate = 0.3, iterations = 1000, print_info = True):
    costs = []
    w = np.zeros((784,10))
    c = 0.0001

    for i in range(iterations):
        x_train_batch, y_train_batch = mnist.train.next_batch(500)
        update = propagation(w, c, x_train_batch, y_train_batch)
        w -= learning_rate * update['dw']

        if i % 100 == 0:
            costs.append(update['cost'])
        
        if i % 100 == 0 and print_info == True:
            print("Iteration " + str(i+1) + " Cost = " + str(update['cost']))
        
    results = {'w':w, 'costs':costs}
    return results

if __name__ == "__main__":    
    results = optimization(learning_rate = 0.3, iterations = 2000, print_info = True)
    w = results['w']
    costs = results['costs']

    x_test_batch, y_test_batch = mnist.train.next_batch(30000)
    y_ = predict(w, x_test_batch)
    accuracy_ = accuracy(y_, y_test_batch)
    
    print("The total accuracy is %f"%(accuracy_))
    results = {
        'w':w,
        'costs':costs,
        'accuracy':accuracy_,
        'y_':y_
    }
    print(results)
