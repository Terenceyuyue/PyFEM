function [np,yp,y] = evaluate(obj,data_x,data_y)
    data_yp = obj.feedforward(data_x);
    [~,yp] = max(data_yp,[],1);
    [~,y] = max(data_y,[],1);
    yp = yp'-1; y = y'-1;
    np = sum(yp==y); % number of correct predictions
end