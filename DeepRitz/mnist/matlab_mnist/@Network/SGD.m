function SGD(obj, training_data_x, training_data_y, epochs, mini_batch_size, eta, varargin)
    n = size(training_data_x,2);
    batch_num = fix(n/mini_batch_size);    % number of mini-batches
    
    err = zeros(batch_num*epochs,1); st = 1;
    for ep = 1:epochs
        kk = randperm(n);  % for shuffling the training data        
        for s = 1:batch_num
            % current mini-batch
            id = kk((s-1)*mini_batch_size+1 : s*mini_batch_size);
            mini_batch_x = training_data_x(:,id);
            mini_batch_y = training_data_y(:,id);

            % feedforward
            [aL,a,z] = obj.feedforward(mini_batch_x);

            % backpropagation
            obj.backprop(mini_batch_y,a,z,eta);
            
            % compute errors
            err(st) = 0.5*mean((aL(:)-mini_batch_y(:)).^2);
            st = st + 1;
        end

        % evaluation of test_data
        if ~isempty(varargin)
            test_data_x = varargin{1};
            test_data_y = varargin{2};
            ntest = size(test_data_x,2);
            np = obj.evaluate(test_data_x,test_data_y);
            fprintf('Epoch %2d :   %d / %d \n', ep, np, ntest);            
        end
    end
    
    plot(err); ylim([0 0.5])
end % end of SGD