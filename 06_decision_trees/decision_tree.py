import numpy as np
import matplotlib.pyplot as plt

class TreeNode():
    def __init__(self, node_type="Leaf", feature=None, threshold=None,
                 value=None, left_branch=None, right_branch=None):
        """
        This class represents a decision node or leaf node in the decision tree.
        A decision node would store the feature index, the threshold value, and 
        a reference to the left and right subtrees. A leaf node only needs to store
        a value.

        Inputs:
        - node_type: (str) Keeps track of the type of note (Decision or Leaf).
        - feature: (int) Index of the feature that is currently being split
        - threshold: (int / float) The threshold value where the split will be made.
        - value: (int / float) Stores the value of a leaf node. 
        - left_branch: (TreeNode) A reference to the left subtree after the split.
        - right_branch: (TreeNode) A reference to the right subtree after the split.
        """
        self.node_type = node_type
        self.feature = feature          
        self.threshold = threshold         
        self.value = value                
        self.left_branch = left_branch      
        self.right_branch = right_branch   
        
class DecisionTree(object):
    def __init__(self, min_samples_split=2, max_depth=np.inf, min_impurity=1e-7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity = min_impurity
    
    def train(self, X, y):
        """
        Builds the decision tree from the training set.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        """

        # Simple check if a feature / dimension is boolean or not
        self.is_bool_feature = np.all(X == X.astype(bool), axis=0)
        # build the tree and remember the root node
        self.root_node = self.create_tree(X, y)

    def create_tree(self, X, y, depth=0):
        """
        Top-down approach for building decision trees (recursive definition).

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.
        - depth: (int) Keeps track of the current depth of the tree

        Output:
        A reference to the current node.
        """
        N, D = X.shape
        
        #########################################################################
        # TODO: Build the tree recursively.                                     #
        #########################################################################
        node = TreeNode()

        if depth < self.max_depth:
            best_gini, best_idx, best_threshold, split_idx = self.choose_best_feature_split(X, y)
            if best_idx is not None:
                indices_left = X[:, best_idx] < best_threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = best_idx
                node.threshold = best_threshold
                node.value = best_gini
                node.left_branch = self._grow_tree(X_left, y_left, depth + 1)
                node.right_branch = self._grow_tree(X_right, y_right, depth + 1)

        return node
        #########################################################################
        #                              END OF YOUR CODE                         #
        #########################################################################  

    def choose_best_feature_split(self, X, y):
        """
        Iterates through all the possible splits and choose the best one according to
        some impurity criteria.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing the ground truth values.

        Output:
        - max_impurity: (float) The maximum impurity gain out of all the splits.
        - best_feature: (int) The index of the feature with maximum impurity gain.
        - best_threshold: (int / float) The value of the best split for the feature.
        - split_idx: (dict)
            - "left_idx": stores the indices of the data in the left subtree after the split
            - "right_idx": stores the indices of the data in the right subtree after the split
        """
        N, D = X.shape
        
        max_impurity = 0
        best_feature = None
        best_threshold = None
        split_idx = {}

        #########################################################################
        # TODO: Choose the best feature to split on.                            #
        #########################################################################
        m = y.size
        if m <= 1:
            return None, None, None, None

        num_parent = [np.sum(y == c) for c in range(len(set(y)))]

        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_threshold = None, None

        for idx in range(D):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * len(set(y))
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(len(set(y))))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(len(set(y))))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        #########################################################################
        #                              END OF YOUR CODE                         #
        ######################################################################### 
        return best_gini, best_idx, best_threshold, split_idx
    
    def traverse_tree(self, X, node):
        """
        Traverse the decision tree to determine the predicted value for a given input.

        Inputs:
        - X: A numpy array of shape (D,) containing training data; The test samples are 
            evaluated one at a time.
        - node: (TreeNode) Current node being evaluated.

        Output:
        Returns the value of the leaf node following the decisions made.
        """

        #########################################################################
        # TODO: Traverse the tree following path based on the decisions that    #
        # made based on the input data.                                         #
        #########################################################################
        if node.node_type == "Leaf":
            return node.value

        X_feature_val = X[node.feature]
        
        if self.is_bool_feature[node.feature]:
            if X_feature_val == node.threshold:
                branch = node.right_branch
            else:
                branch = node.left_branch
        else:
            if X_feature_val >= node.threshold:
                branch = node.right_branch
            else:
                branch = node.left_branch

        return self.traverse_tree(X, branch)
        #########################################################################
        #                              END OF YOUR CODE                         #
        #########################################################################         
    
    def predict(self, X):
        """
        Iterates through each example and traverse the decision tree 
        to get the predicted value.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data

        Output:
        Returns the predicted value following the decisions made in the tree.
        """

        N, D = X.shape
        prediction = []
        
        for i in range(N):
            pred = self.traverse_tree(X[i], node=self.root_node)
            prediction.append(pred)

        return prediction
    
    def compute_leaf(self, y):
        raise NotImplementedError("Should be implemented in the subclass")

    def compute_impurity(self, y_S, y_A, y_B):
        raise NotImplementedError("Should be implemented in the subclass")

    def visualize_tree(self):
        """
        Code for visualizing the tree. This is not part of the exercise but feel free
        to look at the code.
        """
        def traverse_tree_viz(node, parent_pt, center_pt, depth=0):

            if node.node_type == "Decision":
                feature_name = node.feature
                if self.is_bool_feature[node.feature]:
                    text = "F {} == {}".format(feature_name, node.threshold)
                else:
                    text = "F {} >= {}".format(feature_name, node.threshold)
                    
                r = 0.5 - depth * 0.05
                theta = np.pi * 13 / 12 + depth * np.pi / 15
                traverse_tree_viz(node.left_branch, center_pt, (r*np.cos(theta) + center_pt[0], r* np.sin(theta) + center_pt[1]), depth+1)
                theta = - np.pi / 12 - depth * np.pi / 15
                traverse_tree_viz(node.right_branch, center_pt, (r*np.cos(theta) + center_pt[0], r* np.sin(theta) + center_pt[1]), depth+1)

                if parent_pt is None:
                    plotNode(plt_axis, text, center_pt, None, node.node_type)
                else:
                    plotNode(plt_axis, text, center_pt, parent_pt, node.node_type)
                    
            elif node.node_type == "Leaf":
                text = node.value
                plotNode(plt_axis, text, center_pt, parent_pt, node.node_type)
                
        def plotNode(axis, text, center_point, parent_point, node_type):
            decNode = dict(boxstyle="round, pad=0.5", fc='0.8')
            leafNode = dict(boxstyle="round, pad=0.5", fc="0.8")
            arrow_args = dict(arrowstyle="<|-,head_length=0.5,head_width=0.5", edgecolor='black',lw=3, facecolor="black")

            if node_type == "Leaf":
                boxstyle = leafNode
            else:
                boxstyle = decNode

            if parent_point is None:
                axis.text(0.5, 1, text, va="center", ha="center", size=15, bbox=decNode)
            else:
                axis.annotate(text, xy=parent_point, xytext=center_point, 
                    va="center", ha="center", bbox=boxstyle, arrowprops=arrow_args, size=15)

        plt.figure(figsize=(13,15))
        plt_axis = plt.subplot(111, frameon=False)
        traverse_tree_viz(self.root_node, None, (0.5,1))