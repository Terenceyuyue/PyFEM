clc;clear;close all;

%% Load MNIST
% name_data_x, name_data_y, where name = training, validation, test
load mnistdata

%% Parameters
ndim = [784, 15, 10];  % number of neurons on three layers
epochs = 10;
mini_batch_size = 10;
eta = 3;  % learning rate
n = size(training_data_x, 2);  % number of training data
batch_num = fix(n/mini_batch_size);    % number of mini-batches

%% Define activation functions
sigmoid = @(z) 1./(1+exp(-z));
sigmoid_prime = @(z) sigmoid(z).*(1-sigmoid(z));

%% Initialize weights and biases
w2 = randn(ndim([2,1])); % weights from 1-layer to 2-layer
w3 = randn(ndim([3,2]));
b2 = randn(ndim(2),1);   % biases on 2-layer
b3 = randn(ndim(3),1);

%% Train network with SGD
for ep = 1:epochs
    kk = randperm(n);  % for shuffling the training data
    for s = 1:batch_num
        
        % current mini-batch
        id = kk((s-1)*mini_batch_size+1 : s*mini_batch_size);
        mini_batch_x = training_data_x(:,id);
        mini_batch_y = training_data_y(:,id);
        
        % feedforward
        a1 = mini_batch_x;
        z2 = w2*a1 + b2;
        a2 = sigmoid(z2);
        z3 = w3*a2 + b3;
        a3 = sigmoid(z3);
        
        % errors of output neurons
        cost_derivative = a3 - mini_batch_y;
        delta3 = cost_derivative.*sigmoid_prime(z3);
        
        % backpropagation
        delta2 = (w3'*delta3).*sigmoid_prime(z2);
        
        % gradient descent: update weights and biases
        m = size(mini_batch_x,2);
        for i = 1:m   % loops of mini-batch data
            w2 = w2 - eta/m*delta2(:,i)*a1(:,i)';
            b2 = b2 - eta/m*delta2(:,i);
            w3 = w3 - eta/m*delta3(:,i)*a2(:,i)';
            b3 = b3 - eta/m*delta3(:,i);
        end
        
    end
    
    % evaluation of test_data
    a1 = test_data_x;
    z2 = w2*a1 + b2;
    a2 = sigmoid(z2);
    z3 = w3*a2 + b3;
    a3 = sigmoid(z3);
    [~,y_p] = max(a3,[],1);
    [~,y] = max(test_data_y,[],1);
    y_p = y_p'-1; y = y'-1;
    fprintf('Epoch %2d :   %d / %d \n', ep, sum(y_p==y), length(y));
end

%% Recognize handwritten digits
a1 = validation_data_x;
z2 = w2*a1 + b2;
a2 = sigmoid(z2);
z3 = w3*a2 + b3;
a3 = sigmoid(z3);
[~,y_p] = max(a3,[],1);
[~,y] = max(validation_data_y,[],1);
y_p = y_p'-1; y = y'-1;
ratio = sum(sum(y_p==y))/length(y);
fprintf('\n Recognize handwritten digits in validation_data \n');
fprintf(' Accuaracy = %.2f%% \n', ratio*100);
