
#### Libraries
# Standard library
import random
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, ndim):
        self.num_layers = len(ndim)
        self.ndim = ndim
        self.biases = [np.random.randn(s, 1) for s in ndim[1:]]
        self.weights = [np.random.randn(s, t)
                        for s,t in zip(ndim[1:], ndim[:-1])]
    
    # feedforward
    def feedforward(self, data_x):
        a = [];  z = []; 
        a.append(data_x); aL = data_x;
        for b, w in zip(self.biases, self.weights):            
            zs = np.dot(w, aL)+b
            aL = sigmoid(zs);
            z.append(zs);  a.append(aL);
        return aL, a, z
    
    # backpropagation
    def backprop(self,mini_batch_y,a,z,eta):
        # errors of neurons
        delta = [0]*(self.num_layers-1)
        cost_a = self.cost_derivative(a[-1],mini_batch_y)
        delta[-1] = cost_a*sigmoid_prime(z[-1])
        for i in range(len(z)-1):
            w3 = self.weights[-1-i]
            delta3 = delta[-1-i]
            z2 = z[-1-i-1]
            delta[-1-i-1] = np.dot(w3.T,delta3)*sigmoid_prime(z2)
        
        # gradient descent: update weights and biases
        m = mini_batch_y.shape[1]
        for level in range(self.num_layers-1):
            delta2 = delta[level]            
            a1 = a[level]
            w = self.weights[level]
            b = self.biases[level]            
            for i in range(m):
                w = w - eta/m*np.dot(delta2[:,i].reshape(-1,1),a1[:,i].reshape(1,-1))
                b = b - eta/m*delta2[:,i].reshape(-1,1)
            self.weights[level] = w
            self.biases[level] = b
        
    # train network by SGD
    def SGD(self, training_data_x, training_data_y, epochs, mini_batch_size, \
             eta, test_data_x=None, test_data_y=None):
        n = training_data_x.shape[1]
        batch_num = int(n/mini_batch_size)
        
        err = np.zeros(batch_num*epochs,); st = 0;
        for ep in range(epochs):
            kk = random.sample(range(0,n),n)
            for s in range(batch_num):
                # current mini-batch
                id = kk[s*mini_batch_size:(s+1)*mini_batch_size]
                mini_batch_x = training_data_x[:,id]
                mini_batch_y = training_data_y[:,id]
                
                # feedforward
                aL,a,z = self.feedforward(mini_batch_x)
                
                # backpropagation
                self.backprop(mini_batch_y,a,z,eta)
                
                # compute errors
                err[st] = 0.5*np.mean((aL-mini_batch_y)**2)
                st += 1                
                     
                
            if test_data_x.any():
                # evaluation of test_data
                n_correct, _, _ = self.evaluate(test_data_x,test_data_y)
                n_test = test_data_x.shape[1]
                print("Epoch {:2d} : {} / {}".format(ep+1,n_correct,n_test)) 
        
        plt.figure(figsize=(6,4))
        plt.plot(err) 
                

    
    def cost_derivative(self,aL,mini_batch_y):
        return (aL-mini_batch_y)

            
    def evaluate(self, data_x, data_y):
        data_yp, _, _ = self.feedforward(data_x)
        yp = np.argmax(data_yp, axis=0)
        y = np.argmax(data_y,axis=0)   
        num_p = sum(yp == y)
        return num_p, yp, y
    
       

   

#### Activation functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
