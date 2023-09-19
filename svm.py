import numpy as np


class SupportVectorMachine:
    """
     This is a support vector machine using hingeloss and gradient descent
     """

    def __init__(self, num_epochs: int, learning_rate: float, C: float):
        """

        :param num_epochs: count of iterations to run for the stochastic graident descent
        :param learning_rate: a value between 0 and 1 to adjust the weights
        :param C: a regularization term for the slack variables
        """
        self.learning_rate = learning_rate
        self.C = C
        self.num_epochs = num_epochs

    def fit(self, X, y):
        """
        Optimize the weights of the support vector machine based on the feature vectors (X) of shape (num_instances, num_features)
        and the labels y
        :param X: a two dimensional numpy array that contains in the first dimension the instances and in the second the features
        :param y: a one dimension array that contains the labels as 1 and -1
        """
        num_instances, num_features = X.shape
        ones = np.ones((num_instances,1))
        X = np.concatenate((ones, X), axis=1)
        num_features = num_features + 1
        if num_instances:
            w = np.zeros(num_features)
        for epoch in range(self.num_epochs):
            for idx, x in enumerate(X):
                gradient = self.compute_graident(w,x, y[idx])
                w = w - self.learning_rate * gradient
            loss = self.compute_loss(w,X,y)
        self.w = w

    def compute_loss(self, w, X, y):
        """
        Compute the loss of the support vector machine using hinge loss
        :param w: a one dimensionsal array of shape (num_features+1) that contains the weignts in float
        :param X: a two dimensional numpy array that contains in the first dimension the instances and in the second the features
        :param y: a one dimension array that contains the labels as 1 and -1
        :return: the loss which contains 0.5 w*w + hingeloss
        """
        num_instances, _ = X.shape
        predictions = np.dot(X,w)
        distances = 1 - np.multiply(y, predictions )
        distances = np.maximum(0, distances)
        hinge_loss = np.sum(distances)
        return 0.5 * np.dot(w, w) +   self.C * num_instances * hinge_loss


    def compute_graident(self, w, x, y):
        """
        Calculates the gradient of the loss function to update the stochastic gradient descent. The gradient will be either
        w or w + the gradient of the hinge loss
        :param w: a one dimensionsal array of shape (num_features+1) that contains the weignts in float
        :param x: a one dimensional array that represents the current instances for which to calculate the graident
        :param y: the label of the instances
        :return: a one dimensional arry of the shape (num_features+1) that contains the update of the  weignts in float
        """
        if y * np.dot(w,x)<=1:
            gradient =  w  * self.C * y * x
        else:
            gradient =  w
        return gradient

    def predict(self, X):
        """
        Calcualtes the predictions of the support vector machine for a list of instances which are encoded using the
        features in X
        :param X: a two dimensional numpy array that contains in the first dimension the instances and in the second the features
        :return: a one dimensional array of the shape num of instances that is 1 or -1
        """
        num_instances, num_dim = X.shape
        if num_dim == len(self.w) -1:
            ones = np.ones((num_instances,1))
            X = np.concatenate((ones,X),axis=1)
        return np.sign(np.dot(X,self.w))
