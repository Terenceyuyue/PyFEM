function a = sigmoid_prime(z)
a = sigmoid(z).*(1-sigmoid(z));
end