function backprop(obj,mini_batch_y,a,z,eta)
    % errors of neurons
    delta = cell(1,obj.num_layers-1);
    cost_a = cost_derivative(a{end}, mini_batch_y);
    delta{end} = cost_a.*sigmoid_prime(z{end});
    for i = 0:length(z)-2
        w3 = obj.weights{end-i};
        delta3 = delta{end-i};
        z2 = z{end-i-1};
        delta{end-i-1} = (w3'*delta3).*sigmoid_prime(z2);
    end
    % gradient descent: update weights and biases
    m = size(mini_batch_y,2);
    for level = 1:obj.num_layers-1 % 2,...,L
        delta2 = delta{level}; a1 = a{level};
        w = obj.weights{level};
        b = obj.biases{level};
        for i = 1:m   % loops of mini-batch data
            w = w - eta/m*delta2(:,i)*a1(:,i)';
            b = b - eta/m*delta2(:,i);
        end
        obj.weights{level} = w;
        obj.biases{level} = b;
    end

end % end of backprop

% derivative of cost function (w.r.t. a)
function cost_a = cost_derivative(aL, mini_batch_y)
	cost_a = aL - mini_batch_y;
end