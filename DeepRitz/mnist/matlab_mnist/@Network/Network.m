classdef Network < handle
    properties
        num_layers
        ndim
        biases
        weights
    end
    
    methods
        % constructor: initialization
        function obj = Network(ndim)
            obj.num_layers = length(ndim);
            obj.ndim = ndim;
            for s = 1:obj.num_layers-1 % 2,...,L
                obj.biases{s} = randn(ndim(s+1),1);
                obj.weights{s} = randn(ndim([s+1,s]));
            end
        end
        
        % feedforward
        [aL,a,z] = feedforward(obj,data_x);
        
        % backpropagation
        backprop(obj,mini_batch_y,a,z,eta);
        
        % train network by SGD
        SGD(obj, training_data_x, training_data_y, epochs, mini_batch_size, eta, varargin);
        
        % evaluation of test_data
        [np,yp,y] = evaluate(obj,data_x,data_y);
        
    end  % end of methods
    
end