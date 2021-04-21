import numpy as np
from decision_tree import *
        
class ClassificationTree(DecisionTree):
    def __init__(self, min_samples_split=2, max_depth=np.inf,
                 min_impurity=1e-7, impurity_criterion="entropy"):
        DecisionTree.__init__(self, min_samples_split, max_depth, min_impurity)
        self.impurity_criterion = impurity_criterion
        
    def entropy(self, y):
        """
        Compute for Shannon's entropy. It is the expected value (mean) of the information content.
        
        Hint: Look at how to compute log using masked arrays, in order 
              to avoid infinite / NaN (not a number) errors.
        https://stackoverflow.com/questions/21752989/numpy-efficiently-avoid-0s-when-taking-logmatrix
        https://docs.scipy.org/doc/numpy-1.13.0/reference/maskedarray.generic.html
        
        Hint: look up numpy.bincount
        Inputs:
        - y: A numpy array of size (N,).
        """
        #########################################################################
        # TODO: Implement Shannon's entropy                                     #
        #########################################################################
        entropy = None
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return entropy

        
    def information_gain(self,y_S, y_A, y_B):
        """
        Computes for the reduction of information / entropy after the split. 
        Note: Higher entropy means that the distribution of the values are 
        more 'random' (more uniformly distributed)

        Inputs:
        - y_S: A numpy array of size (S,).
        - y_A: A numpy array of size (A,).
        - y_B: A numpy array of size (B,).

        Outputs:
        The information gain of a particular split.
        """

        #########################################################################
        # TODO: Implement Information Gain                                      #
        #########################################################################

        info_gain = None
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return info_gain

    def gini_gain(self, y_S, y_A, y_B):
        """
        Computes for the gain using the gini impurity measure.

        Inputs:
        - y_S: A numpy array of size (S,).
        - y_A: A numpy array of size (A,).
        - y_B: A numpy array of size (B,).

        Outputs:
        The gain, in terms of gini impurity, of a particular split.
        """
        #########################################################################
        # TODO: Implement Gini Gain                                             #
        #########################################################################
        gain = None
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return gain

        
    def gini_impurity(self, y):
        """
        Computes for the gini impurity measure.
        
        Hint: look up numpy.bincount

        Inputs:
        - y: A numpy array of size (N,).

        Outputs:
        Gini impurity measure
        """

        #########################################################################
        # TODO: Implement Gini impurity                                         #
        #########################################################################
        gini = None

        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return gini

    def compute_leaf(self, y):
        """
        This function overrides the compute_leaf of the superclass DecisionTree.

        Determines how the value of the leaf node will be computed.
        For classification, it is simply a majority vote.

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
        split on.

        Inputs:
        - y_S: A numpy array of size (S,).
        - y_A: A numpy array of size (A,).
        - y_B: A numpy array of size (B,).

        Outputs:
        The gain in the impurity measures with respect to the splits.
        """

        if self.impurity_criterion == "entropy":
            return self.information_gain(y_S, y_A, y_B)
        elif self.impurity_criterion == "gini":
            return self.gini_gain(y_S, y_A, y_B)
