clc; clear; close all;

tic;
%% Load MNIST
% name_data_x, name_data_y, where name = training, validation, test
load mnistdata;

%% Parameters
ndim = [784 15 10];
epochs = 10;
mini_batch_size = 10;
eta = 3;

%% Create a Network object
net = Network(ndim);

%% Train network with SGD
net.SGD(training_data_x, training_data_y, epochs, mini_batch_size, eta, ...
        test_data_x, test_data_y);

%% Recognize handwritten digits 
[np,y_p,y] = net.evaluate(validation_data_x,validation_data_y);
ratio = np/length(y);
fprintf('\n Recognize handwritten digits in validation_data \n');
fprintf(' Accuaracy = %.2f%% \n', ratio*100);

toc