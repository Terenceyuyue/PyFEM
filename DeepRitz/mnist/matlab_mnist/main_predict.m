clc;clear;close all;

%% Load MNIST
% name_data_x, name_data_y, where name = training, validation, test
load mnistdata

%% Parameters
ndim = [784, 15, 10];  % number of neurons on three layers
mini_batch_size = 10;

%% Define activation functions
sigmoid = @(z) 1./(1+exp(-z));
sigmoid_prime = @(z) sigmoid(z).*(1-sigmoid(z));

%% Initialize weights and biases
w2 = randn(ndim([2,1])); % weights from 1-layer to 2-layer
w3 = randn(ndim([3,2]));
b2 = randn(ndim(2),1);   % biases on 2-layer
b3 = randn(ndim(3),1);

%% Predict
% current mini-batch
id = 1:mini_batch_size;
mini_batch_x = training_data_x(:,id);
mini_batch_y = training_data_y(:,id);

% feedforward
a1 = mini_batch_x;
z2 = w2*a1 + b2;
a2 = sigmoid(z2);
z3 = w3*a2 + b3;
a3 = sigmoid(z3);
