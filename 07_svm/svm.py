import numpy as np

class SupportVectorMachine(object):
    def __init__(self, C, epsilon=1e-4, max_passes=10, kernel_fn="linear", sigma=0.5):
        """
        Inputs:
        - C: (float) Regularization parameter
        - epsilon: (float) numerical tolerance 
        - max_passes: (int) maximum number of times to iterate over alphas without changing
        - kernel: (str) kernel to be used. (linear or rbf)
        - sigma: (float) parameter of the rbf kernel
        """
        self.C = C
        self.epsilon = epsilon
        self.max_passes = max_passes
        self.kernel_fn = kernel_fn
        self.sigma = sigma
        
    def initialize_parameters(self,N):
        """
        Initialize the parameters of the model to zero.
        
        alpha: (float) Lagrangian multipliers; has shape (N,)
        b: (float) scalar; bias term for the hyperplane 

        Input:
        - N: (int) Number of examples in the data set
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the langrangian multipliers and bias. Technically, the   #
        # weight vector does not need to be initialized since it is computed as     #
        # a function of the x's, alpha's and y's. But it is convenient to have the  #
        # parameters to be all in one collection.                                   #
        #############################################################################
        self.params['alpha'] = None
        self.params['b'] = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        self.params['W'] = 0

    def compute_weights(self,X,y):
        """
        Computes for the weights W. This can be implemented to accomodate both 
        batch and single examples. But for this exercise we do not require you to 
        vectorize your implementations.
        
        Input:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        """
        #############################################################################
        # TODO: Compute for the weights W using the given formula in the notebook.  #
        #############################################################################

        W = None
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################   
        return W
        

    def f(self,x):
        """
        Computes for the hyperplane. This can be implemented to accomodate both 
        batch and single examples. But for this exercise we do not require you to 
        vectorize your implementations.

        Input:
        - x: A numpy array containing a training example; 
        """

        #############################################################################
        # TODO: Compute for the hyperplane f(x).                                    #
        #############################################################################
        f = None
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################  
        return f

    def kernel(self,x, z):
        """
        Computes for the correspoding kernel. This can be implemented to accomodate 
        both batch and single examples. But for this exercise we do not require you 
        to vectorize your implementations.

        Input:
        - x: A numpy array containing a training example; 
        """
        #############################################################################
        # TODO: Implement both the linear and gaussian (rbf) kernels.               #
        #############################################################################

        kernel = None

        if self.kernel_fn == "linear":
            pass
        elif self.kernel_fn == "gaussian" or self.kernel_fn =="rbf":
            pass

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################  
        return kernel

    def train(self, X, y):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        self.X = X
        self.y = y
        N, D = X.shape
        
        self.initialize_parameters(N)
        
        passes = 0
        while passes < self.max_passes:
            alphas_changed = 0
            
            # iterate through all possible alpha_i's
            for i in range(N):   

                self.params["W"] = self.compute_weights(X, y)

                #############################################################################
                # TODO: Compute for the error E_i between the SVM output and the ith class. #
                #############################################################################
                E_i = None
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################  

                # Check for KKT conditions
                if (y[i]*E_i < -self.epsilon and self.params['alpha'][i] < self.C) or \
                        (y[i]*E_i > self.epsilon and self.params['alpha'][i] > 0):

                    #############################################################################
                    # TODO: Randomly choose j such that i is not equal to j.                    #
                    #############################################################################
                    j = None
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################  
                    
                    #############################################################################
                    # TODO: Compute for the error E_i between the SVM output and the ith class. #
                    #############################################################################
                    E_j = None
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 
                    
                    alpha_i = self.params['alpha'][i]
                    alpha_j = self.params['alpha'][j]
                    

                    #############################################################################
                    # TODO: Compute for lower and upper bounds. [L, H]                          #
                    #############################################################################

                    pass

                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 

                    #############################################################################
                    # TODO: Check if the lower bound and upper bound is the same. Note that     #
                    # these are floating values so we only check if they are the same within    #
                    # some numerical precision. If they are the same then we move on to the     #
                    # next alpha_i.                                                             #
                    #############################################################################
                    if None:
                        continue
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    ############################################################################# 

                    
                    #############################################################################
                    # TODO: Compute for eta using the formula given in the notebook.            #
                    #############################################################################
                    eta = None
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # prevent division by zero
                    if eta >= 0:
                        continue
                    #############################################################################
                    # TODO: Compute for new value of alpha_j                                    #
                    #############################################################################
                    new_alpha_j = None
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################
                    
                    #############################################################################
                    # TODO: Clip the values of alpha_j so that it lies within the acceptable    #
                    # bounds.                                                                   #
                    #############################################################################
                    
                    pass

                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    #############################################################################
                    # TODO: Check if the new alpha_j is the same as its old value within some   #
                    # numerical precision.                                                      #
                    #############################################################################
                    if None:
                        continue
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # update the parameter alpha_j
                    self.params['alpha'][j] = new_alpha_j

                    
                    #############################################################################
                    # TODO: Compute for new value of alpha_j                                    #
                    #############################################################################
                    new_alpha_i = None
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################

                    # update the parameter alpha_i
                    self.params['alpha'][i] = new_alpha_i

                    #############################################################################
                    # TODO: Compute for new value of the bias term b.                           #
                    #############################################################################
                    pass
                    #############################################################################
                    #                              END OF YOUR CODE                             #
                    #############################################################################
                    alphas_changed += 1
            if alphas_changed == 0:
                passes += 1
            else:
                passes = 0


        self.params['W'] = self.compute_weights(X, y)

        #############################################################################
        # TODO: Store only the X's, y's, and alpha's that are support vectors       #
        #############################################################################
        support_vectors = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.X = self.X[support_vectors]
        self.y = self.y[support_vectors]
        self.params['alpha'] = self.params['alpha'][support_vectors]
        
    def predict(self, X):
        """
        Predict labels for test data using this classifier. This can be implemented to 
        accomodate both batch and single examples. But for this exercise we do not 
        require you to vectorize your implementations.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        #############################################################################
        # TODO: Compute for the predictions on the given test data.                 #
        #############################################################################
        prediction = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return prediction

