function [aL,a,z] = feedforward(obj,data_x)
    a = cell(1,obj.num_layers);   % a1,...,aL
    z = cell(1,obj.num_layers-1); % z2,...,zL
    a{1} = data_x; aL = data_x;
    for s = 1:obj.num_layers-1
        w = obj.weights{s}; b = obj.biases{s};
        zs = w*aL+b; aL = sigmoid(zs);
        z{s} = zs; a{s+1} = aL;
    end
end