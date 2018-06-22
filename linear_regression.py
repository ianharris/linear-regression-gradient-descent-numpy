import numpy as np
import pandas as pd

class Data():

    def __init__(self, fpath):
        
        # read in the file
        data = pd.read_csv(fpath)

        # assign the features
        self.features = data[['X1', 'X2']].as_matrix()
        
        # get the number of samples
        self.numSamples = np.shape(self.features)[0]

        # assign the labels
        self.labels = data['y'].as_matrix()
        self.labels = np.reshape(self.labels, (self.numSamples, 1))



def hypothesis(theta, features):
 
    m = np.matmul(features, theta[1:, :])
    hyp = np.add(theta[0, :], m)
    return hyp

def cost(theta, features, labels):

    # theta = [1, 3]
    # features = [-1, 2]
    # labels = [-1]
    s = np.subtract(hypothesis(theta, features), labels)
    p = np.power(s, 2.0)
    c = 0.5*np.sum(p)
    return c

def gradient_descent(theta, features, labels, numSamples):

    alpha = np.power(10.0, -2)
    alpha = 2*np.divide(alpha, np.shape(features)[0])

    ntheta = np.matrix([[0.0], [0.0], [0.0]])

    hyp = hypothesis(theta, features)
    
    ntheta[0, :] = theta[0, :] - alpha * np.sum(np.subtract(hyp, labels))
    ntheta[1, :] = theta[1, :] - alpha * np.sum(np.multiply(np.subtract(hyp, labels), np.reshape(features[:, 0], (numSamples, 1))))
    ntheta[2, :] = theta[2, :] - alpha * np.sum(np.multiply(np.subtract(hyp, labels), np.reshape(features[:, 1], (numSamples, 1))))
    
    return ntheta

def main():
    
    # read in the data
    data = Data('data/training-plane.csv')
    
    # initialise theta
    theta = np.subtract(np.multiply(np.random.rand(3, 1), 2), 1)

    # run iterations
    for i in range(2000):
        if i % 100 == 0:
            print('The cost at iteration {} is {}'.format(i, cost(theta, data.features, data.labels)))
        theta = gradient_descent(theta, data.features, data.labels, data.numSamples)

    # calculate a prediction
    y_pred = hypothesis(theta, data.features)
    print('Writing output')
    with open('output.txt', 'w') as f:
        f.write('X1,X2,y,y_pred\n')
        for i in range(np.shape(data.features)[0]):
            f.write('{},{},{},{}\n'.format(data.features[i,0], data.features[i,1], np.asscalar(data.labels[i]), np.asscalar(y_pred[i])))
    
    print(np.transpose(theta))

if __name__ == '__main__':
    main()
