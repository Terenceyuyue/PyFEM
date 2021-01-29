#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
    f.close()
    
    training_data_x = tr_d[0].transpose()
    training_data_y = np.hstack([vectorize_num(i) for i in tr_d[1]])
    validation_data_x = va_d[0].transpose()
    validation_data_y = np.hstack([vectorize_num(i) for i in va_d[1]])
    test_data_x = te_d[0].transpose()
    test_data_y = np.hstack([vectorize_num(i) for i in te_d[1]])
    
    return training_data_x, training_data_y, \
           validation_data_x, validation_data_y, \
           test_data_x, test_data_y

def vectorize_num(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
