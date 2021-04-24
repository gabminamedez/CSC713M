import numpy as np
from decision_tree import *
        
class RegressionTree(DecisionTree):
    def __init__(self, min_samples_split=2, max_depth=np.inf,
                 min_impurity=1e-7):
        DecisionTree.__init__(self, min_samples_split, max_depth, min_impurity)

    def compute_variance_reduction(self, y_S, y_A, y_B):
        """
        For regression the 'gain' is in the form of variance reduction.
        Don't forget to normalize the variances according the their corresponding
        sizes.

        Inputs:
        - y_S: A numpy array of size (S,).
        - y_A: A numpy array of size (A,).
        - y_B: A numpy array of size (B,).

        Outputs:
        Gain in terms of variance reduction.
        """
        #########################################################################
        # TODO: Compute for the reduction in variance of the particular split.  #
        #########################################################################
        var_reduction = None

        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 

        return var_reduction

    def compute_leaf(self, y):
        """
        This function overrides the compute_leaf of the superclass DecisionTree.

        Determines how the value of the leaf node will be computed.
        For regression, it is the mean of the values.

        Inputs:
        - y: A numpy array of size (N,).

        Outputs:
        Value for the leaf node
        """
        #########################################################################
        # TODO: Compute for the resulting value of the leaf node                #
        #########################################################################
        leaf_value = None
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        
        
        return leaf_value

    def compute_impurity(self, y_S, y_A, y_B):
        """
        This function overrides the compute_impurity of the superclass DecisionTree.

        Computes for the appropriate impurity measure for deciding which feature to 
        split on. For regression, we use the squared errors with respect to the output values 
        of the decision tree. Since we use the mean of the values as the output, 
        this is simply the variance.

        Inputs:
        - y_S: A numpy array of size (S,).
        - y_A: A numpy array of size (A,).
        - y_B: A numpy array of size (B,).

        Outputs:
        The gain in the impurity measures with respect to the splits.
        """
        return self.compute_variance_reduction(y_S, y_A, y_B)
